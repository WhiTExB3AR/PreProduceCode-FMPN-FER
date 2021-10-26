import nest_asyncio
from pyngrok import conf, ngrok
import time
# import getpass

# Setting an auth token allows us to open multiple tunnels at the same time
# Ask token
print("Get your authtoken from https://dashboard.ngrok.com/auth")
# authtoken = getpass.getpass()
authtoken = str(input("Enter your authtoken: "))

# ngrok.set_auth_token("1z1di3c8qZMo13cDdSmPlPhuxJP_72JpP4W3UNPXvN4ExUZP2")
ngrok.set_auth_token(authtoken)
ngrok_tunnel = ngrok.connect(8097)
print('==> Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
time.sleep(24*3600)