# 1. Globalna lista wszystkich możliwych kanonicznych urządzeń, które model ma rozpoznawać
CANONICAL_APPLIANCES = [
    "fridge_freezer",
    "washing_machine",
    "dishwasher",
    "microwave",
    "kettle",
    "toaster",
    "tumble_dryer",
    "television",
    "computer_site"
]
# Słownik mapujący ogólne nazwy kolumn na rzeczywiste urządzenia dla poszczególnych domów (np. House 12)
REFIT_CANONICAL_MAPPINGS = {
    "House_12": {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Television Site (Lounge)",
        "Appliance3": "Microwave",
        "Appliance4": "Kettle",
        "Appliance5": "Toaster",
        "Appliance6": "Television Site (Bedroom)",
        "Appliance7": "Not Used",
        "Appliance8": "Not Used",
        "Appliance9": "Not Used"
    },
    "House_2": {
        "Appliance1": "fridge_freezer",
        "Appliance2": "washing_machine",
        "Appliance3": "dishwasher",
        "Appliance4": "television",
        "Appliance5": "microwave",
        "Appliance6": "toaster",
        "Appliance7": "Not Used",
        "Appliance8": "kettle",
        "Appliance9": "Not Used"
    },
    "default": "Not Used"
    # W tym miejscu można łatwo dodać mapowania dla innych domów (np. House_1, House_2 itd.)
}