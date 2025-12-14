import optuna_dashboard

def main():
    optuna_dashboard.run_server("sqlite:///instance/db.sqlite3", host="0.0.0.0", port=5000)
    

if __name__ == "__main__":
    main()
