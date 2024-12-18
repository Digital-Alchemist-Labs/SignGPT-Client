import requests


class SignGPT_API():
  """
  클래스를 통해 반환 값의 재사용성을 높힘
  """

  # variables
  words: str
  result: str
  result_log: str

  sfc_log: str
  cmc_log: str
  spc_log: str

  def __init__(self, name):
    self.name = name

  def sgc(self, words: str):
    """
    입력된 수어 단어를 LLM 서버에 전송하여 결과값을 가져옴
    """
    self.words = words

    response = requests.post(
        "http://0.0.0.0:8000/sgc/invoke/",
        json={'input': {'words': self.words}}
    )

    self.result_log = response.json()
    self.result = self.result_log.get('output').get('content')

    return self.result

  def sgc2(self, words: str):
    """
    입력된 수어 단어를 LLM 서버에 전송하여 결과값을 가져옴
    """
    self.words = words

    self.sfc_log = self.sfc(self.words)
    print(self.sfc_log)

    self.cmc_log = self.cmc(self.sfc_log)
    print(self.cmc_log)

    self.spc_log = self.spc(self.cmc_log)

    print(self.spc_log)

    self.result = self.spc_log

    return self.result

  def sfc(self, words: str):
    """
    문장 완성 채인
    """
    self.words = words

    response = requests.post(
        "http://0.0.0.0:8000/sfc/invoke/",
        json={'input': {'words': self.words}}
    )

    self.result_log = response.json()
    self.result = self.result_log.get('output').get('content')

    return self.result

  def cmc(self, words: str):
    """
    ! 
    """
    self.words = words

    response = requests.post(
        "http://0.0.0.0:8000/cmc/invoke/",
        json={'input': {'question': self.words}}
    )

    self.result_log = response.json()
    self.result = self.result_log.get('output').get('content')

    return self.result

  def ssc(self, words: str):
    """
    챗 모델 채인
    """
    self.words = words

    response = requests.post(
        "http://0.0.0.0:8000/ssc/invoke/",
        json={'input': {'sentence': self.words}}
    )

    self.result_log = response.json()
    self.result = self.result_log.get('output').get('content')

    return self.result
