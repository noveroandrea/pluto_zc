import iio

ctx = iio.Context("ip:127.0.0.1")   # or "ip:localhost"
print("context:", ctx.name)
for dev in ctx.devices:
    print(dev.id, dev.name)