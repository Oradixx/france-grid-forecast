"""GitHub Actions annotations: a failure reason readable on the run page (and through the API)
without opening the logs. Outside Actions, only the logging call happens."""
import logging
import os

log = logging.getLogger(__name__)


def _escape(text, prop=False):
    text = str(text).replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
    return text.replace(':', '%3A').replace(',', '%2C') if prop else text


def annotate(level, message, title=None):
    """level: 'error', 'warning' or 'notice'."""
    getattr(log, 'warning' if level == 'notice' else level)(message)
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        props = f' title={_escape(title, prop=True)}' if title else ''
        print(f'::{level}{props}::{_escape(message)}', flush=True)
