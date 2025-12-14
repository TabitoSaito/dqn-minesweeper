import optuna_dashboard

optuna_dashboard.run_server("sqlite:///instance/db.sqlite3", host="0.0.0.0", port=5000)