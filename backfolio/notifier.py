import requests
from datetime import datetime
from os.path import join, isfile


class AbstractNotifier:
    def __init__(self, name='notifier'):
        self.name = name
        self.session_fields = []

    def reset(self, context=None):
        self.context = context
        return self

    @property
    def current_time(self):
        return self.context.current_time

    def notify(self, _message, _now=None):
        raise NotImplementedError("Notifier must implement `notify()`")

    def formatted_notify(self, message, now=None):
        if now is None or self.context.live_trading():
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = "%12s - [%s]: %s" % (self.name, now, message)
        self.notify(message, now)


class SlackNotifier(AbstractNotifier):
    def __init__(self, name, url):
        super().__init__(name=name)
        self.url = url

    def notify(self, message, _now=None):
        return requests.post(self.url, json={"text": message})


class FileLogger(AbstractNotifier):
    def __init__(self, name):
        super().__init__(name=name)

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.path = join(self.context.session_dir, "%s.log" % self.name)

    def notify(self, message, _now=None):
        mode = 'a' if isfile(self.path) else 'w'
        with open(self.path, mode) as file:
            file.write(message + "\n")
