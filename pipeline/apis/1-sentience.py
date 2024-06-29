#!/usr/bin/env python3
"""Retrieves home planets of all sentient species."""
import requests


def sentientPlanets():
    """Returns list of names of home planets of sentient species."""
    url = "https://swapi.dev/api/species/"
    planets = []
    homeworlds = set()  # To avoid duplicates

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data['results']:
            if species['designation'] == 'sentient' and species['homeworld']:
                homeworlds.add(species['homeworld'])

        url = data['next']

    for homeworld in homeworlds:
        response = requests.get(homeworld)
        planet_data = response.json()
        planets.append(planet_data['name'])

    return planets
