
import datetime
import time as clock


def unix_to_datetime(unix_time: int, micro=False):
    dt = datetime.datetime.fromtimestamp(int(unix_time))
    if not micro:
        formatted_datetime = dt.strftime("%A, %B %d, %Y - %H:%M:%S")
    else:
        formatted_datetime = dt.strftime("%A, %B %d, %Y - %H:%M:%S.%f")
    return formatted_datetime


def date_to_unix_time(day: int, month: int, year: int, days_offset=0):
    try:
        # Get the required date and time
        input_date = datetime.datetime(year, month, day, 0, 0, 0, 0)

        # Subtract the days_offset to get the desired date
        target_date = input_date - datetime.timedelta(days=days_offset)

        # Convert to Unix time (seconds since epoch)
        unix_time = int(target_date.timestamp())
        return unix_time
    except ValueError as e:
        return f"UtilityError: {e}"


def unix_to_datetime_file(unix_time: int):
    dt = datetime.datetime.fromtimestamp(int(unix_time))
    formatted_datetime = dt.strftime("%d-%m-%Y_%H-%M-%S")
    return formatted_datetime


def GetDayStartTime(offset=0):
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Subtract the days_offset to get the desired date
    target_date = current_datetime - datetime.timedelta(days=offset)

    # Set the time to the start of the day (midnight)
    start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Convert to Unix time (seconds since epoch)
    unix_time = int(start_of_day.timestamp())

    return unix_time


def GetWeekStartTime(offset=0):
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Calculate the start of the current week (assuming the week starts on Monday)
    start_of_week = current_datetime - datetime.timedelta(days=current_datetime.weekday())

    # Apply the week offset
    target_week = start_of_week - datetime.timedelta(weeks=offset)

    # Set the time to the start of the day (midnight)
    target_week_start = target_week.replace(hour=0, minute=0, second=0, microsecond=0)

    # Convert to Unix time (seconds since epoch)
    unix_time = int(target_week_start.timestamp())

    return unix_time


def GetMonthStartTime(offset=0):
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Calculate the year and month with the offset
    total_months = current_datetime.year * 12 + current_datetime.month - 1 - offset
    target_year = total_months // 12
    target_month = total_months % 12 + 1

    # Create a datetime object for the start of the target month
    start_of_month = datetime.datetime(target_year, target_month, 1)

    # Convert to Unix time (seconds since epoch)
    unix_time = int(start_of_month.timestamp())

    return unix_time


def GetYearStartTime(offset=0):
    # Get the current time in seconds since the epoch
    current_time = clock.time()

    # Get the current year
    current_year = clock.gmtime(current_time).tm_year

    # Calculate the target year
    target_year = current_year - offset

    # Create a struct_time object for the start of the target year -> *(January 1, 00:00:00)
    start_of_year = clock.struct_time((target_year, 1, 1, 0, 0, 0, 0, 0, -1))

    # Convert the struct_time to Unix time
    unix_time = clock.mktime(start_of_year)
    return unix_time


def eta(time_start, units_done, units_total, text=''):
    time_done = clock.perf_counter() - time_start
    progress = round(units_done / units_total * 100)
    eta_ = round(time_done / units_done * (units_total - units_done))
    print(f"\r{text} | {progress}% | eta={eta_}s | ", end='')
