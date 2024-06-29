import requests

def availableShips(passengerCount):
    """
    Returns the list of ships that can hold a given number of passengers.

    Args:
        passengerCount (int): The number of passengers.

    Returns:
        List[str]: List of ship names that can hold the given number of passengers.
    """
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data['results']:
            passengers = ship['passengers'].replace(',', '')
            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship['name'])

        url = data['next']

    return ships
