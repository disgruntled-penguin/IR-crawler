import urllib3

from spintrace.crawl.crawler import Crawler


class ResetRaw:
    def read(self, *a, **k):
        raise urllib3.exceptions.ProtocolError("Connection broken: reset by peer")


class ResetSession:
    def get(self, *a, **k):
        class R:
            raw = ResetRaw()
            status_code = 200

            def close(self):
                pass
        return R()


def test_connection_reset_while_reading_is_a_failed_fetch():
    c = object.__new__(Crawler)
    c.session = ResetSession()
    c.wait_for = lambda host: None
    logged = []
    c._log_request = lambda *a: logged.append(a)
    assert c.fetch("https://x.test/a", "x.test") == (None, None)
    assert logged and logged[0][-1] == "error=ProtocolError"
