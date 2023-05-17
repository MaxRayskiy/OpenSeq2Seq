#include "scorer.h"

#include <unistd.h>
#include <iostream>

#include "lm/config.hh"
#include "lm/model.hh"
#include "lm/state.hh"
#include "util/string_piece.hh"
#include "util/tokenize_piece.hh"
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_vector.h>
#include <chrono>
#include "decoder_utils.h"
#include <mutex>

using namespace lm::ngram;

Scorer::Scorer(double alpha,
               double beta,
               const std::string& lm_path,
               const std::vector<std::string>& vocab_list,
               const std::string &voc_file_path) {
  this->alpha = alpha;
  this->beta = beta;

  dictionary = nullptr;
  is_character_based_ = true;
  language_model_ = nullptr;

  max_order_ = 0;
  dict_size_ = 0;
  SPACE_ID_ = -1;

  setup(lm_path, vocab_list, voc_file_path);
}

Scorer::~Scorer() {
  if (language_model_ != nullptr) {
    delete static_cast<lm::base::Model*>(language_model_);
  }
  if (dictionary != nullptr) {
    delete static_cast<fst::StdVectorFst*>(dictionary);
  }
}

void Scorer::setup(const std::string& lm_path,
                   const std::vector<std::string>& vocab_list, const std::string &voc_file_path) {
  //if (!voc_file_path.empty()) {}
  // load language model
  load_lm(lm_path);
  // set char map for scorer
  set_char_map(vocab_list);
  // fill the dictionary for FST
  if (!is_character_based()) {
    fill_dictionary_parallel(true);
  }
}

void Scorer::load_lm(const std::string& lm_path) {
  const char* filename = lm_path.c_str();
  VALID_CHECK_EQ(access(filename, F_OK), 0, "Invalid language model path");

  RetriveStrEnumerateVocab enumerate;
  lm::ngram::Config config;
  config.enumerate_vocab = &enumerate;
  language_model_ = lm::ngram::LoadVirtual(filename, config);
  max_order_ = static_cast<lm::base::Model*>(language_model_)->Order();
  vocabulary_ = enumerate.vocabulary;
  for (size_t i = 0; i < vocabulary_.size(); ++i) {
    if (is_character_based_ && vocabulary_[i] != UNK_TOKEN &&
        vocabulary_[i] != START_TOKEN && vocabulary_[i] != END_TOKEN &&
        get_utf8_str_len(enumerate.vocabulary[i]) > 1) {
      is_character_based_ = false;
    }
  }
}

double Scorer::get_log_cond_prob(const std::vector<std::string>& words) {
  lm::base::Model* model = static_cast<lm::base::Model*>(language_model_);
  double cond_prob;
  lm::ngram::State state, tmp_state, out_state;
  // avoid to inserting <s> in begin
  model->NullContextWrite(&state);
  for (size_t i = 0; i < words.size(); ++i) {
    lm::WordIndex word_index = model->BaseVocabulary().Index(words[i]);
    // encounter OOV
    if (word_index == 0) {
      return OOV_SCORE;
    }
    cond_prob = model->BaseScore(&state, word_index, &out_state);
    tmp_state = state;
    state = out_state;
    out_state = tmp_state;
  }
  // return  log10 prob
  return cond_prob;
}

double Scorer::get_sent_log_prob(const std::vector<std::string>& words) {
  std::vector<std::string> sentence;
  if (words.size() == 0) {
    for (size_t i = 0; i < max_order_; ++i) {
      sentence.push_back(START_TOKEN);
    }
  } else {
    for (size_t i = 0; i < max_order_ - 1; ++i) {
      sentence.push_back(START_TOKEN);
    }
    sentence.insert(sentence.end(), words.begin(), words.end());
  }
  sentence.push_back(END_TOKEN);
  return get_log_prob(sentence);
}

double Scorer::get_log_prob(const std::vector<std::string>& words) {
  assert(words.size() > max_order_);
  double score = 0.0;
  for (size_t i = 0; i < words.size() - max_order_ + 1; ++i) {
    std::vector<std::string> ngram(words.begin() + i,
                                   words.begin() + i + max_order_);
    score += get_log_cond_prob(ngram);
  }
  return score;
}

void Scorer::reset_params(float alpha, float beta) {
  this->alpha = alpha;
  this->beta = beta;
}

std::string Scorer::vec2str(const std::vector<int>& input) {
  std::string word;
  for (auto ind : input) {
    word += char_list_[ind];
  }
  return word;
}

std::vector<std::string> Scorer::split_labels(const std::vector<int>& labels) {
  if (labels.empty()) return {};

  std::string s = vec2str(labels);
  std::vector<std::string> words;
  if (is_character_based_) {
    words = split_utf8_str(s);
  } else {
    words = split_str(s, " ");
  }
  return words;
}

void Scorer::set_char_map(const std::vector<std::string>& char_list) {
  char_list_ = char_list;
  char_map_.clear();

  // Set the char map for the FST for spelling correction
  for (size_t i = 0; i < char_list_.size(); i++) {
    if (char_list_[i] == " ") {
      SPACE_ID_ = i;
    }
    // The initial state of FST is state 0, hence the index of chars in
    // the FST should start from 1 to avoid the conflict with the initial
    // state, otherwise wrong decoding results would be given.
    char_map_[char_list_[i]] = i + 1;
  }
}

std::vector<std::string> Scorer::make_ngram(PathTrie* prefix) {
  std::vector<std::string> ngram;
  PathTrie* current_node = prefix;
  PathTrie* new_node = nullptr;

  for (int order = 0; order < max_order_; order++) {
    std::vector<int> prefix_vec;

    if (is_character_based_) {
      new_node = current_node->get_path_vec(prefix_vec, SPACE_ID_, 1);
      current_node = new_node;
    } else {
      new_node = current_node->get_path_vec(prefix_vec, SPACE_ID_);
      current_node = new_node->parent;  // Skipping spaces
    }

    // reconstruct word
    std::string word = vec2str(prefix_vec);
    ngram.push_back(word);

    if (new_node->character == -1) {
      // No more spaces, but still need order
      for (int i = 0; i < max_order_ - order - 1; i++) {
        ngram.push_back(START_TOKEN);
      }
      break;
    }
  }
  std::reverse(ngram.begin(), ngram.end());
  return ngram;
}

void Scorer::fill_dictionary(bool add_space) {
  fst::StdVectorFst dictionary;
  // For each unigram convert to ints and put in trie
  int dict_size = 0;
  for (const auto& word : vocabulary_) {
    bool added = add_word_to_dictionary(
        word, char_map_, add_space, SPACE_ID_ + 1, &dictionary);
    dict_size += added ? 1 : 0;
  }

  dict_size_ = dict_size;

  /* Simplify FST

   * This gets rid of "epsilon" transitions in the FST.
   * These are transitions that don't require a string input to be taken.
   * Getting rid of them is necessary to make the FST determinisitc, but
   * can greatly increase the size of the FST
   */
  fst::RmEpsilon(&dictionary);
  fst::StdVectorFst* new_dict = new fst::StdVectorFst;

  /* This makes the FST deterministic, meaning for any string input there's
   * only one possible state the FST could be in.  It is assumed our
   * dictionary is deterministic when using it.
   * (lest we'd have to check for multiple transitions at each state)
   */
  fst::Determinize(dictionary, new_dict);

  /* Finds the simplest equivalent fst. This is unnecessary but decreases
   * memory usage of the dictionary
   */
  fst::Minimize(new_dict);
  this->dictionary = new_dict;
}

void Scorer::fill_dictionary_parallel(bool add_space) {
    const int num_threads = 10;
    const int chunks = num_threads;
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "fill_dictionary" << std::endl;

    oneapi::tbb::concurrent_vector<fst::StdVectorFst> dictionary_list;
    std::vector<std::vector<std::string>> vocabularies(chunks);

    // split vocabulary into num_threads parts
    int part_size = vocabulary_.size() / chunks;
    for (int i = 0; i < chunks; i++) {
        int start = i * part_size;
        int end = (i + 1) * part_size;
        if (i == num_threads - 1)
            end = vocabulary_.size();

        vocabularies[i] = std::vector<std::string>(vocabulary_.begin() + start, vocabulary_.begin() + end);
    }
    std::mutex mutex;
    auto start_chunks = std::chrono::high_resolution_clock::now();
    oneapi::tbb::parallel_for(
                            oneapi::tbb::blocked_range<size_t>(0, chunks),
                      [&](const oneapi::tbb::blocked_range<size_t> &r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            fst::StdVectorFst dct;
                            for (const auto &word: vocabularies[i]) {
                              add_word_to_dictionary(word, char_map_, add_space, SPACE_ID_ + 1, &dct);
                            }
                            fst::RmEpsilon(&dct);
                            fst::Determinize(dct, &dct);
                            fst::Minimize(&dct);
                            dictionary_list.push_back(dct);
                          }
                      });


    //std::cout << "init size: " << dictionary_list_init.size() << std::endl;
    std::cout << "list size: " << dictionary_list.size() << std::endl;
    auto current = std::chrono::high_resolution_clock::now();
    std::cout << "Union: " << std::chrono::duration_cast<std::chrono::seconds>(current - start_time).count()  << std::endl;
    // Merge the FSTs into a single FST
    fst::StdVectorFst final_fst;
    for (size_t i = 0; i < dictionary_list.size(); ++i) {
        if (i == 0) {
            final_fst = dictionary_list[i];
        } else {
            fst::Union(&final_fst, dictionary_list[i]);
        }
    }

    dict_size_ = final_fst.NumStates();
    current = std::chrono::high_resolution_clock::now();
    std::cout << "RmEpsilon: " << std::chrono::duration_cast<std::chrono::seconds>(current - start_time).count()  << std::endl;
    /* Simplify FST

     * This gets rid of "epsilon" transitions in the FST.
     * These are transitions that don't require a string input to be taken.
     * Getting rid of them is necessary to make the FST determinisitc, but
     * can greatly increase the size of the FST
     */
    fst::RmEpsilon(&final_fst);
    fst::StdVectorFst* new_dict = new fst::StdVectorFst;

    /* This makes the FST deterministic, meaning for any string input there's
     * only one possible state the FST could be in.  It is assumed our
     * dictionary is deterministic when using it.
     * (lest we'd have to check for multiple transitions at each state)
     */
    current = std::chrono::high_resolution_clock::now();
    std::cout << "Determinize: " << std::chrono::duration_cast<std::chrono::seconds>(current - start_time).count()  << std::endl;

    fst::Determinize(final_fst, new_dict);

    /* Finds the simplest equivalent fst. This is unnecessary but decreases
     * memory usage of the dictionary
     */
    current = std::chrono::high_resolution_clock::now();
    std::cout << "Minimize: " << std::chrono::duration_cast<std::chrono::seconds>(current - start_time).count()  << std::endl;
    fst::Minimize(new_dict);
    this->dictionary = new_dict;
    current = std::chrono::high_resolution_clock::now();
    std::cout << "done: " << std::chrono::duration_cast<std::chrono::seconds>(current - start_time).count()  << std::endl;
}
