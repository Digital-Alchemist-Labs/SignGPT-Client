import json
import os
import requests
import cv2
import numpy as np


class TextToSign:
  def __init__(self, mapping_path, url_path, paths_path, paths_path1, mode='url'):
    with open(mapping_path, 'r', encoding='utf-8') as mapping_file:
      self.mapping_data = json.load(mapping_file)

    with open(url_path, 'r', encoding='utf-8') as url_file:
      self.url_data = json.load(url_file)

    with open(paths_path, 'r', encoding='utf-8') as paths_file:
      self.paths_data = json.load(paths_file)

    with open(paths_path1, 'r', encoding='utf-8') as paths_file1:
      self.paths_data1 = json.load(paths_file1)

    self.mode = mode  # 'url' or 'path'

  def find_videos(self, words):
    words_list = [word.strip() for word in words.split(',')]
    video_sources = []

    for word in words_list:
      found = False
      for mapping in self.mapping_data[0]["mapping"]:
        if word in mapping["token"]["kr"]:
          label = mapping["label"]
          if self.mode == 'url':
            for entry in self.url_data["sign-dictionary"]:
              for file, url in entry["file_to_url"].items():
                if label in file:
                  video_sources.append((word, url))
                  found = True
                  break
          elif self.mode == 'path':
            for entry in self.paths_data["sign-dictionary"]:
              for file, path in entry["file_to_path"].items():
                if label in file:
                  video_sources.append((word, path))
                  found = True
                  break
          if found:
            break
      if not found:
        video_sources.append((word, "Not found"))

    return video_sources

  def find_videos1(self, words):
    words_list = [word.strip() for word in words.split(',')]
    video_sources = []

    for word in words_list:
      found = False
      for mapping in self.mapping_data[0]["mapping"]:
        if word in mapping["token"]["kr"]:
          label = mapping["label"]
          if self.mode == 'url':
            for entry in self.url_data["sign-dictionary"]:
              for file, url in entry["file_to_url"].items():
                if label in file:
                  video_sources.append((word, url))
                  found = True
                  break
          elif self.mode == 'path':
            for entry in self.paths_data1["sign-dictionary"]:
              for file, path in entry["file_to_path"].items():
                if label in file:
                  video_sources.append((word, path))
                  found = True
                  break
          if found:
            break
      if not found:
        video_sources.append((word, "Not found"))

    return video_sources

  def display_videos(self, video_sources):
    for word, source in video_sources:
      if source == "Not found":
        print(f"No video found for the word: {word}")
      else:
        print(
            f"Word: {word}, Video {'URL' if self.mode == 'url' else 'Path'}: {source}")

  def display_videos2(self, video_sources):
    for word, source in video_sources:
      if source == "Not found":
        print(f"No video found for the word: {word}")
      else:
        print(
            f"{'Downloading and playing' if self.mode == 'url' else 'Playing'} video for word: {word}")
        if self.mode == 'url':
          video_filename = self.download_video(source)
        else:
          video_filename = source
        if video_filename:
          self.play_video(video_filename)

  def display_videos3(self, video_sources):
    for word, source in video_sources:
      if source == "Not found":
        print(f"No video found for the word: {word}")
      else:
        print(f"Playing video for word: {word}")
        self.play_video_cv2(source)

  def download_video(self, url):
    local_filename = url.split('/')[-1]
    try:
      response = requests.get(url, stream=True)
      if response.status_code == 200:
        with open(local_filename, 'wb') as f:
          for chunk in response.iter_content(chunk_size=8192):
            if chunk:
              f.write(chunk)
        print(f"Downloaded video: {local_filename}")
        return local_filename
      else:
        print(f"Failed to download video: {url}")
        return None
    except Exception as e:
      print(f"An error occurred while downloading the video: {e}")
      return None

  def play_video(self, video_path):
    try:
      cap = cv2.VideoCapture(video_path)
      if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

      cap.release()
      cv2.destroyAllWindows()
    except Exception as e:
      print(f"An error occurred while playing the video: {e}")

  def play_video_cv2(self, video_path):
    try:
      cap = cv2.VideoCapture(video_path)
      if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break

      cap.release()
      cv2.destroyAllWindows()
    except Exception as e:
      print(f"An error occurred while playing the video: {e}")


if __name__ == "__main__":
  module = TextToSign(
      mapping_path="/Users/jaylee_83/Documents/_DigitalAlchemistLabs/git_clones/SignGPT-Client/dictionary/kr-dict-mapping.json",
      url_path="/Users/jaylee_83/Documents/_DigitalAlchemistLabs/git_clones/SignGPT-Client/dictionary/kr-dict-urls.json",
      paths_path="/Users/jaylee_83/Documents/_DigitalAlchemistLabs/git_clones/SignGPT-Client/dictionary/kr-dict-paths.json",
      mode="path"  # Set the mode to 'url' or 'path'
  )

  input_words = "안녕"

  video_sources = module.find_videos(input_words)

  # # Option 1: Display URLs or paths
  # module.display_videos(video_sources)

  # # Option 2: Download and play videos (for URL mode) or play local videos (for path mode)
  # module.display_videos2(video_sources)

  # Option 3: Play videos using OpenCV
  module.display_videos3(video_sources)
  print(video_sources)
