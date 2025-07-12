"""
services.py - Tool implementations for MCP server

© 2025 NextRun Digital. All Rights Reserved.
"""

import datetime
import re
import ast
import operator
import math
import random
from typing import Dict, Any, List, Optional

def get_current_time(timezone=None):
    """Get the current time in a specified timezone"""
    try:
        from datetime import datetime
        import pytz
        
        if timezone:
            try:
                tz = pytz.timezone(timezone)
                current_time = datetime.now(tz)
            except pytz.exceptions.UnknownTimeZoneError:
                return {"error": f"Unknown timezone: {timezone}"}
        else:
            current_time = datetime.now()
            
        return {
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "timezone": timezone or "local"
        }
    except Exception as e:
        return {"error": str(e)}

def calculate_age(birth_date):
    """Calculate age based on birth date"""
    try:
        from datetime import datetime
        
        # Parse birth date (format: YYYY-MM-DD)
        birth_date = datetime.strptime(birth_date, "%Y-%m-%d")
        
        # Calculate age
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        
        return {
            "birth_date": birth_date.strftime("%Y-%m-%d"),
            "age": age,
            "calculation_date": today.strftime("%Y-%m-%d")
        }
    except Exception as e:
        return {"error": str(e)}

def get_weather(location, unit="celsius"):
    """
    Simulate getting weather for a location
    (Note: This is a simulated function for demonstration purposes)
    """
    # For demonstration, we'll return simulated weather data
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Thunderstorm", "Snowy", "Foggy"]
    temp_base = {"celsius": 22, "fahrenheit": 72}
    
    # Pseudo-random but deterministic weather based on location name
    import hashlib
    location_hash = int(hashlib.md5(location.encode()).hexdigest(), 16)
    random.seed(location_hash)
    
    condition = conditions[location_hash % len(conditions)]
    temp_variation = random.randint(-10, 10)
    temperature = temp_base[unit] + temp_variation
    humidity = random.randint(30, 90)
    
    return {
        "location": location,
        "temperature": temperature,
        "unit": unit,
        "condition": condition,
        "humidity": humidity,
        "note": "This is simulated weather data for demonstration purposes"
    }

def search_products(query, category=None, max_results=5):
    """
    Simulate searching for products
    (Note: This is a simulated function for demonstration purposes)
    """
    # Sample product database for demonstration
    product_db = [
        {"id": 1, "name": "Laptop", "category": "Electronics", "price": 999.99, "rating": 4.5},
        {"id": 2, "name": "Smartphone", "category": "Electronics", "price": 699.99, "rating": 4.7},
        {"id": 3, "name": "Headphones", "category": "Electronics", "price": 199.99, "rating": 4.3},
        {"id": 4, "name": "Coffee Maker", "category": "Kitchen", "price": 89.99, "rating": 4.1},
        {"id": 5, "name": "Blender", "category": "Kitchen", "price": 49.99, "rating": 4.0},
        {"id": 6, "name": "Running Shoes", "category": "Sports", "price": 129.99, "rating": 4.6},
        {"id": 7, "name": "Yoga Mat", "category": "Sports", "price": 29.99, "rating": 4.4},
        {"id": 8, "name": "Programming Book", "category": "Books", "price": 39.99, "rating": 4.8},
        {"id": 9, "name": "Novel", "category": "Books", "price": 19.99, "rating": 4.2},
        {"id": 10, "name": "Desk", "category": "Furniture", "price": 249.99, "rating": 4.0},
    ]
    
    # Filter by query (case-insensitive partial match in name)
    results = [p for p in product_db if query.lower() in p["name"].lower()]
    
    # Filter by category if provided
    if category:
        results = [p for p in results if p["category"].lower() == category.lower()]
    
    # Limit results
    results = results[:max_results]
    
    return {
        "query": query,
        "category": category,
        "count": len(results),
        "results": results
    }

def do_math_calculation(expression):
    """
    Safely evaluate a mathematical expression
    """
    # Define allowed operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }
    
    # Define allowed functions
    allowed_functions = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'sqrt': math.sqrt,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'abs': abs,
        'round': round,
    }
    
    def safe_eval(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in operators:
                raise ValueError(f"Unsupported binary operator: {type(node.op)}")
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            return operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in operators:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
            operand = safe_eval(node.operand)
            return operators[type(node.op)](operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are supported")
            if node.func.id not in allowed_functions:
                raise ValueError(f"Function not allowed: {node.func.id}")
            args = [safe_eval(arg) for arg in node.args]
            return allowed_functions[node.func.id](*args)
        elif isinstance(node, ast.Name):
            if node.id == 'pi':
                return math.pi
            elif node.id == 'e':
                return math.e
            raise ValueError(f"Unknown variable: {node.id}")
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")
    
    try:
        # Replace function calls with ast-compatible format
        for func in allowed_functions:
            expression = expression.replace(f"{func}(", f"{func}(")
        
        # Parse expression
        node = ast.parse(expression, mode='eval').body
        result = safe_eval(node)
        
        return {
            "expression": expression,
            "result": result
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e)
        }
