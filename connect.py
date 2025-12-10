import requests    # 用于向目标网站发送请求
username = '221123120334'
password = '2000915lxd'
ip = '10.12.11.149'
url = 'http://192.168.6.1:801/eportal/?c=Portal&a=login&callback=dr1003&login_method=1&user_account=%2C0%2C' + username + '&user_password=' + password +'&wlan_user_ip='+ ip +'&wlan_user_ipv6=&wlan_user_mac=000000000000&wlan_ac_ip=&wlan_ac_name=&jsVersion=3.3.3&v=6210'
response = requests.get(url).status_code  # 直接利用 GET 方式请求这个 URL 同时获取状态码
print("状态码{}".format(response))  # 打印状态码
