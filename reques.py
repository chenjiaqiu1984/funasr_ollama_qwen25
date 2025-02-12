import requests
import time
import datetime
#import itchat

from wxauto import *

# 登录微信

wx=WeChat()
wx.GetSessionList()
friend="zoewz2tt"
#itchat.auto_login(hotReload=True)
#my_friends = itchat.get_friends(update=True)
#supreme1984_user = next((friend['UserName'] for friend in my_friends if friend['NickName'] == 'zoe52666'), None)

def send_msg_to_single_user( msg, user):
    try:
        print(f"向用户`{user}`发送消息:{msg}")
        wx.ChatWith(user)  # 打开`对方`聊天窗口
        wx.SendMsg(msg)
        print("发送完毕")
    except Exception as e:
        print("发送失败，原因:", e)


def send_wechat_message(message):
    """
    发送微信消息
    """

    print("send message")
    #if supreme1984_user:
        #itchat.send(message, toUserName=supreme1984_user)

def check_appointment(time_str):
    """
    检查预约情况
    """
    url = "https://api-college.xinli001.com/listener/student/appointmentDateDetail"
    params = {
        "time": time_str,
        "canAppointment": 0,
        "page": 1,
        "pageSize": 10,
        "userId": 1008222326,
        "tenantId": 2
    }
    response = requests.get(url, params=params)
    data = response.json()
    print(data)
    if data['code'] == 1 and data['data']['interviewList']['totalCount'] > 0:
        for interview in data['data']['interviewList']['list']:
            remaining_num = interview['remainingNum']

            if remaining_num != 0:
                current_time = datetime.datetime.now()
                send_msg_to_single_user(f"当前时间：{current_time}，剩余预约名额：{remaining_num}，时间：{interview['interviewStartTime']}，请前往 https://static.xinli001.com/msite/index.html#/qingting-interview/intervieweeHome?tenantId=2  预约",friend)
                return

def main():
    # 获取今天日期
    today = datetime.date.today()
    # 计算未来两周的日期
    end_date = today + datetime.timedelta(weeks=2)
    # 循环检查未来两周的每一天
    current_time = datetime.datetime.now()
    round = 1
    #send_msg_to_single_user(f"当前时间：{current_time}，预约系统监控开始，开始时间：{today}，结束时间：{end_date}", friend)
    while today <= end_date:
        # 格式化日期为字符串



        current_time = datetime.datetime.now()
        #send_msg_to_single_user(f"当前时间：{current_time}，预约系统监控开始， 轮次：{round}",
            #friend)
        today = datetime.date.today()

        inital_date=today.strftime('%Y-%m-%d')

        today += datetime.timedelta(days=2)
        time_str = today.strftime('%Y-%m-%d')
        # 检查预约情况
        check_appointment(time_str)
        # 每5分钟检查一次
        time.sleep(1)
        # 移到下一天
        today += datetime.timedelta(days=1)
        print(time_str,inital_date)

        time_str = today.strftime('%Y-%m-%d')
        # 检查预约情况
        check_appointment(time_str)
        # 每5分钟检查一次
        time.sleep(1)
        # 移到下一天
        today += datetime.timedelta(days=1)
        print(time_str, inital_date)

        time_str = today.strftime('%Y-%m-%d')
        # 检查预约情况
        check_appointment(time_str)
        # 每5分钟检查一次
        time.sleep(1)
        # 移到下一天
        today += datetime.timedelta(days=1)
        print(time_str, inital_date)

        time_str = today.strftime('%Y-%m-%d')
        # 检查预约情况
        check_appointment(time_str)
        # 每5分钟检查一次
        time.sleep(1)
        # 移到下一天
        today += datetime.timedelta(days=1)
        print(time_str, inital_date)

        time_str = today.strftime('%Y-%m-%d')
        # 检查预约情况
        check_appointment(time_str)
        # 每5分钟检查一次
        time.sleep(1)
        # 移到下一天
        today += datetime.timedelta(days=1)
        print(time_str, inital_date)

        time_str = today.strftime('%Y-%m-%d')
        # 检查预约情况
        check_appointment(time_str)
        # 每5分钟检查一次
        time.sleep(1)
        # 移到下一天
        today += datetime.timedelta(days=1)
        print(time_str, inital_date)

        time_str = today.strftime('%Y-%m-%d')
        # 检查预约情况
        check_appointment(time_str)
        # 每5分钟检查一次
        time.sleep(1)
        # 移到下一天
        today += datetime.timedelta(days=1)
        print(time_str, inital_date)

        time_str = today.strftime('%Y-%m-%d')
        # 检查预约情况
        check_appointment(time_str)
        # 每5分钟检查一次
        time.sleep(1)
        # 移到下一天
        today += datetime.timedelta(days=1)
        print(time_str, inital_date)

        time_str = today.strftime('%Y-%m-%d')
        # 检查预约情况
        check_appointment(time_str)
        # 每5分钟检查一次
        time.sleep(1)
        # 移到下一天
        today += datetime.timedelta(days=1)
        print(time_str, inital_date)

        time_str = today.strftime('%Y-%m-%d')
        # 检查预约情况
        check_appointment(time_str)
        # 每5分钟检查一次
        time.sleep(1)
        # 移到下一天

        print(time_str, inital_date)

        time.sleep(120)
        round+=1



if __name__ == "__main__":
    main()