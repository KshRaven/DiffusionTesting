
from numba import njit, objmode
from colorama import Fore, Style

import colorama
import datetime


def cmod(text: str, color: colorama.Fore = Fore.GREEN, style: colorama.Style = Style.BRIGHT):
    text: str = color + style + str(text) + Style.RESET_ALL
    return text


@njit()
def formatted_time(time_done: int, color=Fore.LIGHTGREEN_EX, style=Style.BRIGHT):
    time_done = round(time_done, 1)
    with objmode(out='unicode_type'):
        delta = datetime.timedelta(seconds=time_done)
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        td = "{:02d}:{:02d}:{:05.2f}".format(hours, minutes, seconds + delta.microseconds / 1e6)
        out = f"{cmod(td, color, style)}"
    return out


@njit()
def progress_print(ticks_done, ticks_total, time_done, mult=1, extra=''):
    progress = round(ticks_done / ticks_total * 100, 1)
    tpt = time_done * mult / ticks_done
    eta = (ticks_total - ticks_done) * tpt
    with objmode():
        eta = formatted_time(eta)
        print(f"\rProgress = {progress}% eta={eta}{extra}", end='')
