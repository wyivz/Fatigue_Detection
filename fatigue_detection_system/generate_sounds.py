import numpy as np
from scipy.io.wavfile import write
import os
import subprocess

# 创建声音目录（如果不存在）
sounds_dir = os.path.join('static', 'sounds')
os.makedirs(sounds_dir, exist_ok=True)

def generate_beep(freq, duration, sample_rate=44100, volume=0.5):
    """生成一个指定频率和持续时间的蜂鸣声"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return (np.sin(freq * 2 * np.pi * t) * volume).astype(np.float32)

def generate_sweep(start_freq, end_freq, duration, sample_rate=44100, volume=0.5):
    """生成一个从起始频率扫到结束频率的音调"""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # 使用对数扫频
    freq = np.exp(np.linspace(np.log(start_freq), np.log(end_freq), len(t)))
    return (np.sin(2 * np.pi * freq * t / sample_rate) * volume).astype(np.float32)

def generate_complex_alert(sample_rate=44100):
    """生成一个基本的报警声音"""
    # 基本警报：交替的高低音调
    sound = np.array([], dtype=np.float32)
    
    # 添加3个高低交替的蜂鸣声
    for _ in range(3):
        high_beep = generate_beep(880, 0.2, sample_rate, 0.7)  # 高音调 (A5)
        low_beep = generate_beep(440, 0.2, sample_rate, 0.7)   # 低音调 (A4)
        silence = np.zeros(int(0.1 * sample_rate), dtype=np.float32)  # 短暂停顿
        
        sound = np.concatenate([sound, high_beep, silence, low_beep, silence])
    
    return sound

def generate_fatigue_alert(sample_rate=44100):
    """生成疲劳警报声音"""
    # 疲劳警报：逐渐加快的蜂鸣声，然后是扫频
    sound = np.array([], dtype=np.float32)
    
    # 逐渐加快的蜂鸣声
    for gap in [0.3, 0.25, 0.2, 0.15, 0.1]:
        beep = generate_beep(660, 0.2, sample_rate, 0.7)  # E5音
        silence = np.zeros(int(gap * sample_rate), dtype=np.float32)
        sound = np.concatenate([sound, beep, silence])
    
    # 添加一个长的警告扫频
    sweep = generate_sweep(330, 880, 1.0, sample_rate, 0.8)
    sound = np.concatenate([sound, sweep])
    
    return sound

def generate_phone_alert(sample_rate=44100):
    """生成打电话警报声音"""
    # 打电话警报：电话铃声类似的声音
    sound = np.array([], dtype=np.float32)
    
    # 类似电话铃声的双音
    for _ in range(3):
        tone1 = generate_beep(1318.5, 0.4, sample_rate, 0.6)  # E6
        tone2 = generate_beep(1567.9, 0.4, sample_rate, 0.6)  # G6
        silence = np.zeros(int(0.4 * sample_rate), dtype=np.float32)
        
        sound = np.concatenate([sound, tone1, tone2, silence])
    
    return sound

def generate_smoking_alert(sample_rate=44100):
    """生成抽烟警报声音"""
    # 抽烟警报：短促的声音
    sound = np.array([], dtype=np.float32)
    
    # 连续4个短促下降音调
    for _ in range(4):
        sweep_down = generate_sweep(880, 330, 0.2, sample_rate, 0.7)
        silence = np.zeros(int(0.1 * sample_rate), dtype=np.float32)
        sound = np.concatenate([sound, sweep_down, silence])
    
    # 添加最后一个长警告音
    final_beep = generate_beep(554.4, 0.8, sample_rate, 0.8)  # C#5
    sound = np.concatenate([sound, final_beep])
    
    return sound

def save_wav_and_convert_to_mp3(filename, data, sample_rate):
    """保存WAV文件并转换为MP3"""
    wav_path = os.path.join(sounds_dir, f"{filename}.wav")
    mp3_path = os.path.join(sounds_dir, f"{filename}.mp3")
    
    # 保存为WAV
    write(wav_path, sample_rate, data)
    print(f"已保存: {wav_path}")
    
    # 如果系统支持，尝试转换为MP3（需要ffmpeg）
    try:
        subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-b:a', '192k', mp3_path], 
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"已转换为MP3: {mp3_path}")
        
        # 转换成功后删除WAV文件
        os.remove(wav_path)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"转换为MP3失败: {e}")
        print("注意: MP3转换需要安装ffmpeg。由于转换失败，将保留WAV文件作为替代。")

def create_placeholder_file(filename, content="This is a placeholder audio file."):
    """创建占位文件"""
    file_path = os.path.join(sounds_dir, filename)
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"已创建占位文件: {file_path}")

# 生成并保存所有声音文件
print("正在生成声音文件...")

# 尝试使用scipy生成wav文件，然后转换为mp3
try:
    # 基本警报
    alert_sound = generate_complex_alert()
    save_wav_and_convert_to_mp3("alert", alert_sound, 44100)
    
    # 疲劳警报
    fatigue_sound = generate_fatigue_alert()
    save_wav_and_convert_to_mp3("fatigue_alert", fatigue_sound, 44100)
    
    # 打电话警报
    phone_sound = generate_phone_alert()
    save_wav_and_convert_to_mp3("phone_alert", phone_sound, 44100)
    
    # 抽烟警报
    smoking_sound = generate_smoking_alert()
    save_wav_and_convert_to_mp3("smoking_alert", smoking_sound, 44100)
    
except ImportError as e:
    print(f"警告: 无法生成声音文件: {e}")
    print("将创建占位文件代替...")
    
    # 创建占位文件
    create_placeholder_file("alert.txt", "请替换为实际的alert.mp3文件")
    create_placeholder_file("fatigue_alert.txt", "请替换为实际的fatigue_alert.mp3文件")
    create_placeholder_file("phone_alert.txt", "请替换为实际的phone_alert.mp3文件")
    create_placeholder_file("smoking_alert.txt", "请替换为实际的smoking_alert.mp3文件")

print("声音文件生成完成！") 