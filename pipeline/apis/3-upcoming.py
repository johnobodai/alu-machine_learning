#!/usr/bin/env python3
"""
Retrieve and display information about the upcoming SpaceX launch using the SpaceX API.
"""
import requests
from datetime import datetime, timezone

def get_upcoming_launch():
    """
    Retrieve information about the upcoming SpaceX launch from the SpaceX API.

    Returns:
        str: Formatted string with launch details.
    """
    api_url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        launches = response.json()
        if launches:
            # Sort launches by date_unix and get the earliest one
            upcoming_launch = min(launches, key=lambda x: x['date_unix'])
            
            # Extract information
            launch_name = upcoming_launch['name']
            launch_date_utc = upcoming_launch['date_utc']
            rocket_name = upcoming_launch['rocket']
            launchpad_name = upcoming_launch['launchpad']['name']
            launchpad_locality = upcoming_launch['launchpad']['locality']
            
            # Convert UTC date to local time
            launch_date_local = convert_utc_to_local(launch_date_utc)
            
            # Format output string
            formatted_output = "{} ({}) {} - {} ({})".format(
                launch_name, launch_date_local, rocket_name, launchpad_name, launchpad_locality
            )
            
            return formatted_output
        else:
            return "No upcoming launches found"
    else:
        return "Error fetching data from SpaceX API"

def convert_utc_to_local(utc_datetime_str):
    """
    Convert UTC datetime string to local time.

    Args:
        utc_datetime_str (str): UTC datetime string in ISO format.

    Returns:
        str: Local datetime string in '%Y-%m-%dT%H:%M:%S%z' format.
    """
    utc_datetime = datetime.fromisoformat(utc_datetime_str.replace('Z', '+00:00'))
    local_datetime = utc_datetime.astimezone()
    return local_datetime.strftime('%Y-%m-%dT%H:%M:%S%z')
