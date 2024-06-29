#!/usr/bin/env python3
"""
Retrieve names of home planets of all sentient species from SWAPI.
"""
import requests

def sentientPlanets():
    """
    Retrieves the list of names of home planets of all sentient species.
    Returns a list of planet names.
    """
    planets = []
    url = "https://swapi.dev/api/species/"
    
    while url:
        response = requests.get(url)
        data = response.json()
        
        for species in data['results']:
            if species['designation'] == 'sentient' and species['homeworld']:
                homeworld_url = species['homeworld']
                homeworld_response = requests.get(homeworld_url)
                if homeworld_response.status_code == 200:
                    homeworld_data = homeworld_response.json()
                    planets.append(homeworld_data['name'])
                else:
                    planets.append('unknown')
        
        url = data['next']
    
    return planets
