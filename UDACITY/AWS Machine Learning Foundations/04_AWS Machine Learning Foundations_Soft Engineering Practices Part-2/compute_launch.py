def days_until_launch(current_day, launch_day):
    """"Returns the days left before launch.
    
    current_day (int) - current day in integer
    launch_day (int) - launch day in integer
    """

    # return launch_day - current_day

    remaining_day = launch_day - current_day
    if remaining_day >=0:
        return remaining_day
    else:
        return 0