import gymnasium as gym
from stable_baselines3 import PPO
import time
import numpy as np

# 加载保存的模型（使用zip文件）
print("正在加载模型：walker2d_ppo_model.zip...")
model = PPO.load("walker2d_ppo_model.zip")
print("模型加载成功！")

# 创建一个环境实例
print("创建Walker2d-v5环境...")
env = gym.make("Walker2d-v5", render_mode="human")

# 运行5次测试循环，使用同一个环境
for test_run in range(5):
    print(f"\n========================")
    print(f"开始第 {test_run + 1}/5 次测试")
    print(f"========================")
    
    # 重置环境（不关闭环境）
    print("重置环境...")
    obs, _ = env.reset()
    
    # 记录每次运行的总奖励
    total_reward = 0
    
    # 记录开始时间
    start_time = time.time()
    
    print("开始测试循环...")
    step_count = 0
    terminated = False
    truncated = False
    
    # 运行直到episode结束
    while not (terminated or truncated):
        # 使用模型预测动作
        action, _ = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 累加奖励
        total_reward += reward
        step_count += 1
        
        # 可选：每100步显示一次进度
        if step_count % 100 == 0:
            print(f"已完成 {step_count} 步，当前累计奖励：{total_reward:.2f}")
    
    # 计算运行时间
    elapsed_time = time.time() - start_time
    
    print(f"\n测试 {test_run + 1} 完成:")
    print(f"总步数: {step_count}")
    print(f"总奖励: {total_reward:.2f}")
    print(f"运行时间: {elapsed_time:.2f} 秒")
    
    # 在每次运行之间暂停一下，但不关闭环境
    if test_run < 4:  # 如果不是最后一次运行
        print("等待2秒后开始下一次测试...")
        time.sleep(2)

# 最后才关闭环境
env.close()
print("\n所有测试运行完成！环境已关闭")