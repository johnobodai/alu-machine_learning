#!/usr/bin/env python3
"""
Retrieve and print the location of a GitHub user from the GitHub API.
"""
import requests
import sys
import time

def get_user_location(api_url):
    """
    Retrieves the location of a GitHub user from the GitHub API.

    Args:
        api_url (str): The full API URL of the GitHub user.

    Returns:
        str: The location of the GitHub user.
    """
    headers = {'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(api_url, headers=headers)
    
    if response.status_code == 404:
        return "Not found"
    elif response.status_code == 403:
        reset_time = int(response.headers['X-Ratelimit-Reset'])
        current_time = int(time.time())
        minutes_until_reset = max(1, (reset_time - current_time) // 60)
        return "Reset in {} min".format(minutes_until_reset)
    elif response.status_code == 200:
        user_data = response.json()
        if 'location' in user_data and user_data['location']:
            return user_data['location']
        else:
            return "Location not provided"
    else:
        return "Unexpected error"
