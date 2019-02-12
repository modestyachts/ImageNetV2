import base64

from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


class AESCipher(object):
    def __init__(self, key, iv):
        self.key = key
        self.iv  = iv

    def encrypt(self, raw):
        raw = pad(raw, AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
        return base64.urlsafe_b64encode(cipher.encrypt(raw))

    def decrypt(self, enc):
        enc = base64.urlsafe_b64decode(enc)
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv)
        return unpad(cipher.decrypt(enc), AES.block_size)

