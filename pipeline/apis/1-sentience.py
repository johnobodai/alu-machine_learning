#!/usr/bin/env python3
"""Retrieves home planets of all sentient species."""
import requests

def sentientPlanets():
    """Returns list of names of home planets of sentient species."""
    url = "https://swapi.dev/api/species/"
    planets = []
    homeworlds = set()  # To avoid duplicates
    planets_not_found = []

    while url:
        response = requests.get(url)
        data = response.json()
        
        for species in data['results']:
            if species['designation'] == 'sentient' and species['homeworld']:
                homeworlds.add(species['homeworld'])
        
        url = data['next']
    
    for homeworld in homeworlds:
        response = requests.get(homeworld)
        if response.status_code == 200:
            planet_data = response.json()
            planets.append(planet_data['name'])
        else:
            planets_not_found.append(homeworld)
    
    if planets_not_found:
        return f"Planets not found: {planets_not_found}"
    else:
        return "OK"


