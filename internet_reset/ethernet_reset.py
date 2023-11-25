import subprocess
import time

def disable_ethernet():
    # tắt kết nối Ethernet
    subprocess.run(["ipconfig", "/release"])
    print("Đã tắt kết nối Ethernet.")

def enable_ethernet():
    # bật kết nối Ethernet
    subprocess.run(["ipconfig", "/renew"])
    print("Đã bật kết nối Ethernet.")

disable_ethernet()

# Đợi 5 giây trước khi bật lại kết nối Ethernet
time.sleep(5)

enable_ethernet()
