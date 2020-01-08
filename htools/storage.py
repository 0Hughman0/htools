import datetime

def valid_date(name):
    if len(name) < 10:
        return False    
    try:
        datetime.datetime.strptime(name[:10], '%Y-%m-%d')
        return True
    except ValueError as e:
        return False
    