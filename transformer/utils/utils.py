from datetime import datetime

def verbose(func):
    """
    A decorator that adds verbosity to the function it wraps.

    Parameters
    ----------
    func : function
        The function to wrap.

    Returns
    -------
    function
        A new function that wraps the original function and adds verbosity.

    """
    def wrapper(*args, **kwargs):
        print(f"Entering {func.__name__} with args: {args}, kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"Exiting {func.__name__} with result: {result}")
        return result
    
    return wrapper

def timer(func):
    """
    A decorator that adds verbosity to the function it wraps.

    Parameters
    ----------
    func : function
        The function to wrap.

    Returns
    -------
    function
        A new function that wraps the original function and adds execution time.

    """
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        print(f"{func.__name__} Duration: {end-start}")
        return result
    
    return wrapper