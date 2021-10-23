import nest_asyncio
from pyngrok import conf, ngrok
import time

# Setting an auth token allows us to open multiple
# tunnels at the same time
ngrok.set_auth_token("1z1di3c8qZMo13cDdSmPlPhuxJP_72JpP4W3UNPXvN4ExUZP2")
ngrok_tunnel = ngrok.connect(8097)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
time.sleep(24*3600)