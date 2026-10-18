"""Arranca el dashboard en local: python correr_local.py [puerto]"""
import sys

from app import app

app.run(host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8050, debug=False)
