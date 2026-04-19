import logging
import requests
import secrets
import hashlib
import time
from urllib.parse import urlencode
from threading import Lock
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class CodeForcesAPI:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        rate_limit: float = 2.0,
        proxy: Optional[dict] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limit = rate_limit
        self.last_req_time = 0
        self.lock = Lock()
        self.session = requests.Session()
        if proxy:
            self.session.proxies.update(proxy)

    def request(
        self,
        method: str,
        params: dict[str, str] | None = {},
        requires_auth: bool = False,
        timeout: float | None = 5,
        retry: int | None = 0,
    ):
        try:
            local_params = dict(params) if params is not None else {}

            with self.lock:
                if requires_auth:
                    # https://codeforces.com/apiHelp
                    local_params["apiKey"] = self.api_key
                    local_params["time"] = str(int(time.time()))
                    rand = "".join(secrets.choice("0123456789") for _ in range(6))
                    s = f"{rand}/{method}?{urlencode(sorted(local_params.items()))}#{self.api_secret}"
                    local_params["apiSig"] = (
                        rand + hashlib.sha512(bytes(s, encoding="utf8")).hexdigest()
                    )

                delay = self.rate_limit - (time.time() - self.last_req_time)
                if delay > 0:
                    time.sleep(delay)
                self.last_req_time = time.time()

            url = "https://codeforces.com/api/" + method
            res = self.session.get(url, params=local_params, timeout=timeout).json()

            if res["status"] == "FAILED":
                raise Exception("CodeForces API request failed: " + res["comment"])
            return res["result"]

        except Exception as e:
            logger.error(f"Error occurred while making API request: {e}")
            if retry and retry > 0:
                logger.info(f"Retrying API request... ({retry} retries left)")
                return self.request(method, params, requires_auth, timeout, retry - 1)
            else:
                raise Exception(f"CodeForces API request failed: {e}")
