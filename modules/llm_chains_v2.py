import requests
from typing import Dict, Any


class SignGPT_API:
  """
  A class to enhance reusability of return values for sign language processing.
  """

  def __init__(self, base_url: str):
    self.base_url = base_url
    self.words: str = ""
    self.result: str = ""
    self.result_log: Dict[str, Any] = {}

  def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a POST request to the specified endpoint.
    """
    try:
      response = requests.post(
          f"{self.base_url}/{endpoint}/invoke/",
          json={'input': data}
      )
      response.raise_for_status()
      return response.json()
    except requests.RequestException as e:
      raise Exception(f"API request failed: {str(e)}")

  def sgc(self, words: str) -> str:
    """
    Send input sign language words to the LLM server and retrieve the result.
    """
    self.words = words
    self.result_log = self._make_request("sgc", {'words': self.words})
    self.result = self.result_log.get('output', {}).get('content', '')
    return self.result

  def sgc2(self, words: str) -> str:
    """
    Process input sign language words through multiple chains.
    """
    self.words = words
    self.sfc_result = self.sfc(self.words)
    print(self.sfc_result)
    self.cmc_result = self.cmc(self.sfc_result)
    print(self.cmc_result)
    self.ssc_result = self.ssc(self.cmc_result)
    print(self.ssc_result)
    self.result = self.ssc_result
    return self.result

  def sfc(self, words: str) -> str:
    """
    Sentence completion chain.
    """
    self.words = words
    self.result_log = self._make_request("sfc", {'words': self.words})
    self.result = self.result_log.get('output', {}).get('content', '')
    return self.result

  def cmc(self, words: str) -> str:
    """
    Content modification chain.
    """
    self.words = words
    self.result_log = self._make_request("cmc", {'question': self.words})
    self.result = self.result_log.get('output', {}).get('content', '')
    return self.result

  def ssc(self, words: str) -> str:
    """
    Chat model chain.
    """
    self.words = words
    self.result_log = self._make_request("ssc", {'sentence': self.words})
    self.result = self.result_log.get('output', {}).get('content', '')
    return self.result


if __name__ == "__main__":
  test = SignGPT_API(base_url="http://0.0.0.0:8000")

  print(test.sgc2("안녕하세요"))
  print(test.result_log)
