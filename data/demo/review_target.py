"""User management module — contains intentional bugs for demo."""

import json
import os

def load_users(filepath):
    # Bug 1: No file existence check
    f = open(filepath, 'r')
    data = json.load(f)
    # Bug 2: File handle never closed (no context manager)
    return data

def get_user_by_email(users, email):
    # Bug 3: Case-sensitive comparison without normalization
    for user in users:
        if user['email'] == email:
            return user
    return None

def calculate_discount(price, discount_percent):
    # Bug 4: No input validation, division possible issues
    return price * discount_percent / 100

def process_batch(items):
    results = []
    for i in range(len(items)):
        # Bug 5: Modifying list while iterating conceptually wrong pattern
        item = items[i]
        if item.get('status') == 'active':
            result = item['value'] * 1.1
            results.append(result)
        # Bug 6: Silent skip of non-active items with no logging
    return results

API_KEY = "sk-1234567890abcdef"  # Bug 7: Hardcoded secret

def connect_to_db(host, port):
    # Bug 8: No error handling, no timeout
    import socket
    s = socket.socket()
    s.connect((host, port))
    return s

def parse_config(config_str):
    # Bug 9: Using eval on external input
    return eval(config_str)

class UserCache:
    cache = {}  # Bug 10: Mutable class variable shared across instances

    def __init__(self):
        pass

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value
