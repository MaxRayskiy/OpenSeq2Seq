# Copyright (c) 2018 NVIDIA Corporation

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import pandas as pd
import tensorflow as tf
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io.wavfile import write
import librosa

from .encoder_decoder import EncoderDecoderModel
from open_seq2seq.utils.utils import deco_print
from StringIO import StringIO


def plot_spectrogram_w_target(
    ground_truth,
    generated_sample,
    post_net_sample,
    attention,
    target_sample,
    target,
    audio_length,
    logdir,
    train_step,
    number=0,
    append=False,
    vmin=None,
    vmax=None,
    save_to_tensorboard=False
):

  ground_truth = ground_truth.astype(float)
  generated_sample = generated_sample.astype(float)
  post_net_sample = post_net_sample.astype(float)
  attention = attention.astype(float)
  target_sample = target_sample.astype(float)
  target = target.astype(float)

  fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, figsize=(8, 15))

  if vmin is None:
    vmin = min(
        np.min(ground_truth), np.min(generated_sample), np.min(post_net_sample)
    )
  if vmax is None:
    vmax = max(
        np.max(ground_truth), np.max(generated_sample), np.max(post_net_sample)
    )

  colour1 = ax1.imshow(
      ground_truth.T,
      cmap='viridis',
      interpolation=None,
      aspect='auto',
      vmin=vmin,
      vmax=vmax
  )
  colour2 = ax2.imshow(
      generated_sample.T,
      cmap='viridis',
      interpolation=None,
      aspect='auto',
      vmin=vmin,
      vmax=vmax
  )
  colour3 = ax3.imshow(
      post_net_sample.T,
      cmap='viridis',
      interpolation=None,
      aspect='auto',
      vmin=vmin,
      vmax=vmax
  )
  colour4 = ax4.plot(target_sample, 'g.')
  colour4 = ax4.plot(target, 'r.')
  ax4.axvline(x=audio_length)
  colour5 = ax5.imshow(
      attention.T, cmap='viridis', interpolation=None, aspect='auto'
  )

  ax1.invert_yaxis()
  ax1.set_ylabel('fourier components')
  ax1.set_title('training data')

  ax2.invert_yaxis()
  ax2.set_ylabel('fourier components')
  ax2.set_title('decoder results')

  ax3.invert_yaxis()
  ax3.set_ylabel('fourier components')
  ax3.set_title('post net results')

  ax4.set_title('Stop Token Prediction')

  ax5.invert_yaxis()
  ax5.set_title('attention')
  ax5.set_ylabel('inputs')

  plt.xlabel('time')

  ax1.axis('off')
  ax2.axis('off')
  ax3.axis('off')
  ax4.axis('off')
  ax5.axis('off')

  fig.subplots_adjust(right=0.8)
  cbar_ax1 = fig.add_axes([0.85, 0.45, 0.05, 0.45])
  fig.colorbar(colour1, cax=cbar_ax1)
  cbar_ax3 = fig.add_axes([0.85, 0.1, 0.05, 0.14])
  fig.colorbar(colour5, cax=cbar_ax3)

  if save_to_tensorboard:
    tag = "{}_image".format(append)
    s = StringIO()
    fig.savefig(s, dpi=300)
    summary = tf.Summary.Image(
        encoded_image_string=s.getvalue(),
        height=int(fig.get_figheight() * 300),
        width=int(fig.get_figwidth() * 300)
    )
    summary = tf.Summary.Value(tag=tag, image=summary)
    plt.close(fig)

    return summary

  else:
    if append:
      name = '{}/Output_step{}_{}_{}.png'.format(
          logdir, train_step, number, append
      )
    else:
      name = '{}/Output_step{}_{}.png'.format(logdir, train_step, number)
    if logdir[0] != '/':
      name = "./" + name
    #save
    fig.savefig(name, dpi=300)
    plt.close(fig)
    return None


def plot_spectrograms(
    specs,
    titles,
    target_sample,
    audio_length,
    logdir,
    train_step,
    number=0,
    append=False,
    vmin=None,
    vmax=None,
    save_to_tensorboard=False
):
  num_figs = len(specs) + 1
  fig, ax = plt.subplots(nrows=num_figs, figsize=(8, num_figs * 3))

  figures = []
  for i, (spec, title) in enumerate(zip(specs, titles)):
    spec = spec.astype(float)
    colour = ax[i].imshow(
        spec.T, cmap='viridis', interpolation=None, aspect='auto'
    )
    figures.append(colour)
    ax[i].invert_yaxis()
    ax[i].set_title(title)
    ax[i].axis('off')
    fig.colorbar(colour, ax=ax[i])
  target_sample = target_sample.astype(float)
  target_fig = ax[-1].plot(target_sample, 'g.')
  ax[-1].axvline(x=audio_length)

  plt.xlabel('time')

  if save_to_tensorboard:
    tag = "{}_image".format(append)
    s = StringIO()
    fig.savefig(s, dpi=300)
    summary = tf.Summary.Image(
        encoded_image_string=s.getvalue(),
        height=int(fig.get_figheight() * 300),
        width=int(fig.get_figwidth() * 300)
    )
    summary = tf.Summary.Value(tag=tag, image=summary)
    plt.close(fig)

    return summary
  else:
    if append:
      name = '{}/Output_step{}_{}_{}.png'.format(
          logdir, train_step, number, append
      )
    else:
      name = '{}/Output_step{}_{}.png'.format(logdir, train_step, number)
    if logdir[0] != '/':
      name = "./" + name
    #save
    fig.savefig(name, dpi=300)

    plt.close(fig)
    return None


def save_audio(
    magnitudes, logdir, step, mode="train", number=0, save_to_tensorboard=False
):
  signal = griffin_lim(magnitudes.T**1.2)
  if save_to_tensorboard:
    tag = "{}_audio".format(mode)
    s = StringIO()
    write(s, 22050, signal)
    summary = tf.Summary.Audio(encoded_audio_string=s.getvalue())
    summary = tf.Summary.Value(tag=tag, audio=summary)
    return summary
  else:
    file_name = '{}/sample_step{}_{}_{}.wav'.format(logdir, step, number, mode)
    if logdir[0] != '/':
      file_name = "./" + file_name
    write(file_name, 22050, signal)
    return None


def griffin_lim(magnitudes, n_iters=50):
  """
  PARAMS
  ------
  magnitudes: spectrogram magnitudes
  stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
  """

  phase = np.exp(2j * np.pi * np.random.rand(*magnitudes.shape))
  complex_spec = magnitudes * phase
  signal = librosa.istft(complex_spec)

  for i in range(n_iters):
    _, phase = librosa.magphase(librosa.stft(signal, n_fft=1024))
    complex_spec = magnitudes * phase
    signal = librosa.istft(complex_spec)
  return signal


class Text2Speech(EncoderDecoderModel):

  @staticmethod
  def get_required_params():
    return dict(
        EncoderDecoderModel.get_required_params(), **{
            'save_to_tensorboard': bool,
        }
    )

  def __init__(self, params, mode="train", hvd=None):
    super(Text2Speech, self).__init__(params, mode=mode, hvd=hvd)
    self.save_to_tensorboard = self.params["save_to_tensorboard"]

  def _create_decoder(self):
    self.params['decoder_params']['num_audio_features'] = (
        self.get_data_layer().params['num_audio_features']
    )
    return super(Text2Speech, self)._create_decoder()

  def maybe_print_logs(self, input_values, output_values, training_step):
    dict_to_log = {}
    step = training_step
    spec, target, _ = input_values['target_tensors']
    predicted_decoder_spectrograms = output_values[0]
    predicted_final_spectrograms = output_values[1]
    attention_mask = output_values[2]
    target_output = output_values[3]
    y_sample = spec[0]
    target = target[0]
    predicted_spectrogram = predicted_decoder_spectrograms[0]
    predicted_final_spectrogram = predicted_final_spectrograms[0]
    attention_mask = attention_mask[0]
    target_output = target_output[0]
    audio_length = output_values[4][0]

    im_summary = plot_spectrogram_w_target(
        y_sample,
        predicted_spectrogram,
        predicted_final_spectrogram,
        attention_mask,
        target_output,
        target,
        audio_length,
        self.params["logdir"],
        step,
        append="train",
        save_to_tensorboard=self.save_to_tensorboard
    )
    dict_to_log['image'] = im_summary

    predicted_final_spectrogram = predicted_final_spectrogram[:audio_length -
                                                              1, :]
    predicted_final_spectrogram = self.get_data_layer(
    ).get_magnitude_spec(predicted_final_spectrogram)
    wav_summary = save_audio(
        predicted_final_spectrogram,
        self.params["logdir"],
        step,
        save_to_tensorboard=self.save_to_tensorboard
    )
    dict_to_log['audio'] = wav_summary

    if self.save_to_tensorboard:
      return dict_to_log
    else:
      return {}

  def finalize_evaluation(self, results_per_batch, training_step):
    dict_to_log = {}
    step = training_step
    sample = results_per_batch[-1]
    input_values = sample[0]
    output_values = sample[1]
    y_sample, target = input_values['target_tensors']
    predicted_spectrogram = output_values[0]
    predicted_final_spectrogram = output_values[1]
    attention_mask_sample = output_values[2]
    target_output_sample = output_values[3]
    audio_length = output_values[4]

    im_summary = plot_spectrogram_w_target(
        y_sample,
        predicted_spectrogram,
        predicted_final_spectrogram,
        attention_mask_sample,
        target_output_sample,
        target,
        audio_length,
        self.params["logdir"],
        step,
        append="eval",
        save_to_tensorboard=self.save_to_tensorboard
    )
    dict_to_log['image'] = im_summary

    if audio_length > 2:
      predicted_final_spectrogram = predicted_final_spectrogram[:audio_length -
                                                                1, :]

      predicted_final_spectrogram = self.get_data_layer(
      ).get_magnitude_spec(predicted_final_spectrogram)
      wav_summary = save_audio(
          predicted_final_spectrogram,
          self.params["logdir"],
          step,
          mode="eval",
          save_to_tensorboard=self.save_to_tensorboard
      )
      dict_to_log['audio'] = wav_summary

    if self.save_to_tensorboard:
      return dict_to_log
    else:
      return {}

  def evaluate(self, input_values, output_values):
    # Need to reduce amount of data sent for horovod
    output_values = [item[-3] for item in output_values]
    input_values = {
        key: [value[0][-3], value[1][-3]] for key, value in input_values.items()
    }
    return [input_values, output_values]

  def infer(self, input_values, output_values):
    if self.on_horovod:
      raise ValueError('Inference is not supported on horovod')
    return [input_values, output_values]

  def finalize_inference(self, results_per_batch, output_file):
    print("output_file is ignored for ts2")
    print("results are logged to the logdir")
    dict_to_log = {}
    batch_size = len(results_per_batch[0][0]['source_tensors'][0])
    for i, sample in enumerate(results_per_batch):
      input_values = sample[0]
      output_values = sample[1]
      predicted_final_spectrograms = output_values[1]
      attention_mask = output_values[2]
      stop_tokens = output_values[3]
      sequence_lengths = output_values[4]

      for j in range(len(predicted_final_spectrograms)):
        predicted_final_spectrogram = predicted_final_spectrograms[j]
        attention_mask_sample = attention_mask[j]
        stop_tokens_sample = stop_tokens[j]

        specs = [predicted_final_spectrogram, attention_mask_sample]
        titles = ["final spectrogram", "attention"]
        audio_length = sequence_lengths[j]

        if "mel" in self.get_data_layer().params['output_type']:
          mag_spec = self.get_data_layer(
          ).inverse_mel(predicted_final_spectrogram)
          log_mag_spec = np.log(np.clip(mag_spec, a_min=1e-5, a_max=None))
          specs.append(log_mag_spec)
          titles.append("linear spectrogram")

        im_summary = plot_spectrograms(
            specs,
            titles,
            stop_tokens_sample,
            audio_length,
            self.params["logdir"],
            0,
            number=i * batch_size + j,
            append="infer"
        )
        dict_to_log['image'] = im_summary

        if audio_length > 2:
          predicted_final_spectrogram = predicted_final_spectrogram[:
                                                                    audio_length
                                                                    - 1, :]
          predicted_final_spectrogram = self.get_data_layer(
          ).get_magnitude_spec(predicted_final_spectrogram)
          wav_summary = save_audio(
              predicted_final_spectrogram,
              self.params["logdir"],
              0,
              mode="infer",
              number=i * batch_size + j,
              save_to_tensorboard=False
          )

          dict_to_log['audio'] = wav_summary

    return {}
