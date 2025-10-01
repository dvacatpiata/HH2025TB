from dash_app_tb import app

# Expose the Dash application instance for deployment
server = app.server

if __name__ == "__main__":
    app.run_server()
