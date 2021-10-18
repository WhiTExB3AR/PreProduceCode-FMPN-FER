import nest_asyncio
from pyngrok import ngrok
import time

ngrok_tunnel = ngrok.connect(8097)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
time.sleep(24*3600)