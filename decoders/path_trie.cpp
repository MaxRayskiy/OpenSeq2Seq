#include "path_trie.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_vector.h>

#include <iostream>
#include "decoder_utils.h"

PathTrie::PathTrie() {
  log_prob_b_prev = -NUM_FLT_INF;
  log_prob_nb_prev = -NUM_FLT_INF;
  log_prob_b_cur = -NUM_FLT_INF;
  log_prob_nb_cur = -NUM_FLT_INF;
  score = -NUM_FLT_INF;

  ROOT_ = -1;
  character = ROOT_;
  exists_ = true;
  parent = nullptr;

  dictionary_ = nullptr;
  dictionary_state_ = 0;
  has_dictionary_ = false;
  offset = 0;

  matcher_ = nullptr;
}

PathTrie::~PathTrie() {
  for (auto child : children_) {
    delete child.second;
  }
}

PathTrie* PathTrie::get_path_trie(int new_char, bool reset) {
  auto child = children_.begin();
  for (child = children_.begin(); child != children_.end(); ++child) {
    if (child->first == new_char) {
      break;
    }
  }
  if (child != children_.end()) {
    if (!child->second->exists_) {
      child->second->exists_ = true;
      child->second->log_prob_b_prev = -NUM_FLT_INF;
      child->second->log_prob_nb_prev = -NUM_FLT_INF;
      child->second->log_prob_b_cur = -NUM_FLT_INF;
      child->second->log_prob_nb_cur = -NUM_FLT_INF;
    }
    return (child->second);
  } else {
    if (has_dictionary_) {
      matcher_->SetState(dictionary_state_);
      bool found = matcher_->Find(new_char + 1);
      if (!found) {
        // Adding this character causes word outside dictionary
        auto FSTZERO = fst::TropicalWeight::Zero();
        auto final_weight = dictionary_->Final(dictionary_state_);
        bool is_final = (final_weight != FSTZERO);
        if (is_final && reset) {
          dictionary_state_ = dictionary_->Start();
        }
        return nullptr;
      } else {
        PathTrie* new_path = new PathTrie;
        new_path->character = new_char;
        new_path->parent = this;
        new_path->dictionary_ = dictionary_;
        new_path->dictionary_state_ = matcher_->Value().nextstate;
        new_path->has_dictionary_ = true;
        new_path->matcher_ = matcher_;
        children_.push_back(std::make_pair(new_char, new_path));
        return new_path;
      }
    } else {
      PathTrie* new_path = new PathTrie;
      new_path->character = new_char;
      new_path->parent = this;
      children_.push_back(std::make_pair(new_char, new_path));
      return new_path;
    }
  }
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output, std::vector<uint32_t>* timestamps) {
  return get_path_vec(output, ROOT_, std::numeric_limits<size_t>::max(), timestamps);
}

PathTrie* PathTrie::get_path_vec(std::vector<int>& output,
                                 int stop,
                                 size_t max_steps,
                                 std::vector<uint32_t>* timestamps) {
  if (character == stop || character == ROOT_ || output.size() == max_steps) {
    std::reverse(output.begin(), output.end());
    if (timestamps) {
      std::reverse(timestamps->begin(), timestamps->end());
    }
    return this;
  } else {
    output.push_back(character);
    if (timestamps) {
      if (timestamps->size() == 0 || output[output.size()-1] == 0 || parent->character == ROOT_ || parent->character == 0) {
        timestamps->push_back(offset);
      }
    }
    return parent->get_path_vec(output, stop, max_steps, timestamps);
  }
}

void PathTrie::iterate_to_vec(std::vector<PathTrie*>& output) {
  if (exists_) {
    log_prob_b_prev = log_prob_b_cur;
    log_prob_nb_prev = log_prob_nb_cur;

    log_prob_b_cur = -NUM_FLT_INF;
    log_prob_nb_cur = -NUM_FLT_INF;

    score = log_sum_exp(log_prob_b_prev, log_prob_nb_prev);
    output.push_back(this);
  }
  for (auto child : children_) {
    child.second->iterate_to_vec(output);
  }
}

// same as iterate_to_vec but without recursion
void PathTrie::iterate_to_vec_no_rec(std::vector<PathTrie*>& output) {
  std::deque<PathTrie*> deque;
  deque.push_back(this);
  if ( this->process_element() ) {
    output.push_back(this);
  }

  while(!deque.empty()) {
    PathTrie* element = deque.front();
    deque.pop_front();
    std::vector<std::pair<int, PathTrie*>> children;
    element->get_children(children);

    for (auto child : children) {
      if ( child.second->process_element() ) {
        output.push_back(child.second);
      }
      std::vector<std::pair<int, PathTrie*>> grandchildren;
      child.second->get_children(grandchildren);
      for (auto grandchild : grandchildren) {
        deque.push_back(grandchild.second);
      }
    }
  }

}

bool PathTrie::process_element() {
  if (exists_) {
    log_prob_b_prev = log_prob_b_cur;
    log_prob_nb_prev = log_prob_nb_cur;

    log_prob_b_cur = -NUM_FLT_INF;
    log_prob_nb_cur = -NUM_FLT_INF;

    score = log_sum_exp(log_prob_b_prev, log_prob_nb_prev);
    return true;
  } else {
    return false;
  }
}

// starts parallel update for all children of the root (current node)
void PathTrie::iterate_to_vec_from_root(std::vector<PathTrie*>& output) {
  if ( this->process_element() ) {
    output.push_back(this);
  }
  std::vector<std::pair<int, PathTrie*>> children;
  this->get_children(children);

  oneapi::tbb::concurrent_vector<std::vector<PathTrie*>> concurr_output;
  concurr_output.reserve(children.size());


  oneapi::tbb::parallel_for(
          oneapi::tbb::blocked_range<size_t>(0, children.size()),
          [&](const oneapi::tbb::blocked_range<size_t> &r) {
              std::vector<PathTrie*> local_output;
              for (size_t i = r.begin(); i != r.end(); ++i) {
                  children[i].second->iterate_to_vec_no_rec(local_output);
              }
              concurr_output.push_back(std::move(local_output));
          });

  for (auto& vec : concurr_output) {
    output.insert(output.end(), vec.begin(), vec.end());
  }

}

void PathTrie::remove() {
  exists_ = false;

  if (children_.size() == 0) {
    auto child = parent->children_.begin();
    for (child = parent->children_.begin(); child != parent->children_.end();
         ++child) {
      if (child->first == character) {
        parent->children_.erase(child);
        break;
      }
    }

    if (parent->children_.size() == 0 && !parent->exists_) {
      parent->remove();
    }

    delete this;
  }
}

void PathTrie::set_dictionary(fst::StdVectorFst* dictionary) {
  dictionary_ = dictionary;
  dictionary_state_ = dictionary->Start();
  has_dictionary_ = true;
}

using FSTMATCH = fst::SortedMatcher<fst::StdVectorFst>;
void PathTrie::set_matcher(std::shared_ptr<FSTMATCH> matcher) {
  matcher_ = matcher;
}
