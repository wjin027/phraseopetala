import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import colorsys
import math
import pyttsx3
import speech_recognition as sr
import threading
from textblob import TextBlob
import zhipuai
import time
from wordcloud import WordCloud
import jieba
import os

# 初始化语音引擎
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# 初始化语音识别器
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# 初始化智谱 AI
zhipuai.api_key = "API"

def analyze_sentiment_with_glm(text):
    """使用智谱 AI 的 ChatGLM 分析文本情感"""
    try:
        response = zhipuai.model_api.invoke(
            model="chatglm_turbo",
            prompt=[
                {"role": "user", "content": f"请分析以下文本的情感倾向。如果是安慰、鼓励、赞美的话语请回复positive；如果是批评、贬低、否定、生气、失望的话语请回复negative；其他中性的话语回复neutral。只需回复：positive、negative 或 neutral。文本：{text}"}
            ],
            temperature=0.3,
        )
        
        if isinstance(response, dict):
            if response.get("code") == 200:
                content = response["data"]["choices"][0]["content"].strip().lower()
                content = content.replace("积极", "positive").replace("消极", "negative").replace("中性", "neutral")
                if any(word in content for word in ['positive', 'negative', 'neutral']):
                    for word in ['positive', 'negative', 'neutral']:
                        if word in content:
                            print("ChatGLM分析结果:", word)
                            return word
                return 'neutral'
            else:
                print(f"ChatGLM API 调用失败: {response.get('msg', '未知错误')}")
                return None
        else:
            print(f"ChatGLM API 返回格式错误")
            return None
    except Exception as e:
        print(f"ChatGLM API 调用错误: {e}")
        return None

def analyze_sentiment(text):
    """分析文本情感"""
    if not text.strip():
        return None
    
    try:
        return analyze_sentiment_with_glm(text)
    except Exception as e:
        print("ChatGLM分析失败")
        return None

def save_speech_to_file(text, sentiment):
    """将识别到的语音文本和情感保存到文件"""
    with open("speech_records.txt", "a", encoding="utf-8") as file:
        file.write(f"{text} [{sentiment}]\n")

class FloatingText:
    def __init__(self, text, x, y, color, lifetime=2.0, fade_duration=1.5):
        self.text = text
        self.x = x
        self.y = y
        self.color = color
        self.lifetime = lifetime  # 总持续时间保持2秒
        self.fade_duration = fade_duration  # 淡出时间延长到1.5秒
        self.birth_time = time.time()
        self.alpha = 255
        self.vy = -0.5  # 降低上升速度使文字更容易读取
        
    def update(self):
        """更新文字状态，返回是否仍然活跃"""
        age = time.time() - self.birth_time
        if age > self.lifetime:
            return False
        
        # 更新位置和透明度
        self.y += self.vy
        
        # 在生命周期最后fade_duration秒开始淡出
        if age > (self.lifetime - self.fade_duration):
            # 计算淡出进度
            fade_progress = (age - (self.lifetime - self.fade_duration)) / self.fade_duration
            self.alpha = int(255 * (1 - fade_progress))
        return True

    def draw(self, draw, font_path):
        """绘制文字"""
        try:
            from PIL import ImageFont
            # 使用较小的字体大小来保持像素感
            font_size = 16
            
            # 尝试加载像素字体，如果失败则使用默认字体
            try:
                font = ImageFont.truetype('C:/Windows/Fonts/PIXEARG_.TTF', font_size)  # Pixel Arial 11
            except:
                try:
                    font = ImageFont.truetype('C:/Windows/Fonts/PressStart2P-Regular.ttf', font_size)  # Press Start 2P
                except:
                    font = ImageFont.truetype(font_path, font_size)
            
            # 创建半透明黑色
            color = (0, 0, 0, self.alpha)
            
            # 获取文本大小
            bbox = draw.textbbox((0, 0), self.text, font=font)
            text_width = bbox[2] - bbox[0]
            
            # 绘制文字，确保居中
            x = self.x - text_width // 2
            draw.text((x, self.y), self.text, fill=color, font=font)
            
        except Exception as e:
            print(f"绘制文字时出错: {e}")

class Particle:
    def __init__(self, x, y, colors, velocity_range=(-2, 2), lifetime=3.0):
        self.x = float(x)
        self.y = float(y)
        self.color = random.choice(colors)
        self.vx = random.uniform(*velocity_range)
        self.vy = random.uniform(*velocity_range)
        self.lifetime = lifetime
        self.birth_time = time.time()
        self.alpha = 255  # 透明度
        self.size = 4  # 粒子大小
        
    def update(self):
        self.x += self.vx
        self.y += self.vy
        # 添加重力效果
        self.vy += 0.1
        # 计算透明度
        age = time.time() - self.birth_time
        self.alpha = int(255 * (1 - age / self.lifetime))
        return age < self.lifetime

    def draw(self, draw):
        if self.alpha > 0:
            # 将十六进制颜色转换为RGB
            r = int(self.color[1:3], 16)
            g = int(self.color[3:5], 16)
            b = int(self.color[5:7], 16)
            # 创建带透明度的颜色
            color = (r, g, b, self.alpha)
            # 绘制4x4像素的粒子
            x, y = int(self.x), int(self.y)
            draw.rectangle([x, y, x + self.size - 1, y + self.size - 1], fill=color)

class PixelPlant:
    def __init__(self, pixel_size=4):
        self.pixel_size = pixel_size
        # 调整基础尺寸为3072:1920的比例
        self.base_height = 480  # 基础高度
        self.base_width = int(self.base_height * (3072/1920))  # 保持3072:1920的比例
        self.width = self.base_width
        self.height = self.base_height
        self.grid_width = self.width // pixel_size
        self.grid_height = self.height // pixel_size

        # 添加花朵计数和生长状态
        self.flower_count = 0
        self.growth_stage = 1
        self.ready_to_grow = False
        self.last_branch_y = None  # 记录最后一个分支的位置
        self.force_branch = False  # 新添加：标记是否强制生长侧芽
        self.is_growing = False  # 新增标志位，用于确保一次只生长一处

        # 添加词云背景相关属性
        self.wordcloud_background = None
        self.last_file_size = 0  # 记录文件大小以检测变化
        self.speech_file_path = "C:/Users/vincc/Documents/program/creativeCoding/finalPrototype/speech_records.txt"

        # 像素艺术调色板
        self.palettes = {
            'earth': ['#8B4513', '#A0522D', '#6B4423'],  # 土色
            'pot': ['#8B4513', '#A0522D', '#D2691E'],    # 花盆色
            'stem': ['#228B22', '#32CD32', '#90EE90', '#98FB98'],   # 茎干色
            'flower': [
                ['#FF69B4', '#FFB6C1', '#FFC0CB', '#FFF0F5'],  # 粉色花
                ['#9370DB', '#8A2BE2', '#9400D3', '#E6E6FA'],  # 紫色花
                ['#FFD700', '#FFA500', '#FF8C00', '#FFEFD5']   # 橙色花
            ],
            'leaf': ['#228B22', '#32CD32', '#90EE90', '#98FB98', '#F0FFF0']  # 叶子色
        }

        # 存储植物相关的像素状态
        self.pot_pixels = {}
        self.stem_pixels = {}
        self.flower_pixels = {}
        self.leaf_pixels = {}

        # 为花朵预先选择一个配色方案
        self.selected_flower_palette = random.choice(self.palettes['flower'])

        # 初始化植物
        self.parts = []  # 存储所有部件
        self.max_parts = 20  # 最大部件数量
        self.butterflies = []  # 存储蝴蝶信息
        self._initialize_pot_and_stem()

        # 初始化时直接添加一个蝴蝶
        self._add_butterfly()

        self.is_listening = True
        self.speech_thread = None
        self.last_growth_time = time.time()
        self.growth_cooldown = 2  # 生长冷却时间（秒）

        # 添加粒子系统
        self.particles = []
        self.growth_colors = ['#FFD700', '#FFFF00', '#32CD32', '#90EE90']  # 黄色和绿色
        self.wither_colors = ['#FF0000', '#8B0000', '#800080', '#4B0082']  # 红色和紫色

        # 添加浮动文字列表
        self.floating_texts = []
        # 添加字体路径
        self.font_path = self._get_font_path()

    def _initialize_pot_and_stem(self):
        """初始化并存储花盆和茎干的像素状态"""
        # 将花盆放在画面中央偏下的位置
        center_x = self.grid_width // 2
        # 初始化花盆
        pot_height = 20
        pot_width = 24
        
        # 计算花盆的垂直位置，确保在画面下方
        pot_bottom = self.grid_height  # 距离底部留出一些空间
        
        # 存储花盆像素
        for y in range(pot_bottom - pot_height, pot_bottom):
            width = int(pot_width * (1 - (pot_bottom - y) / pot_height * 0.2))
            for x in range(center_x - width, center_x + width):
                self.pot_pixels[(x, y)] = random.choice(self.palettes['pot'])

        # 存储花盆装饰（土壤）
        soil_start_y = pot_bottom - pot_height + 1
        for y in range(soil_start_y, soil_start_y + 6):
            for x in range(center_x - pot_width + 4, center_x + pot_width - 4):
                if random.random() < 0.8:
                    self.pot_pixels[(x, y)] = random.choice(self.palettes['earth'])

        # 存储茎干像素
        stem_height = 50
        start_y = pot_bottom - pot_height + 2
        
        # 主茎（更自然的弯曲）
        current_x = center_x
        prev_dx = 0
        for y in range(start_y - stem_height, start_y):
            # 添加平滑的弯曲
            if random.random() < 0.3:
                dx = random.choice([-1, 0, 1])
                if dx * prev_dx < 0:  # 如果方向相反
                    dx = 0
                prev_dx = dx
                current_x += dx
            
            # 使用渐变色
            color_idx = int((y - (start_y - stem_height)) / stem_height * (len(self.palettes['stem']) - 1))
            color = self.palettes['stem'][color_idx]
            self.stem_pixels[(current_x, y)] = color
            
            # 随机添加分枝
            if random.random() < 0.15:
                branch_length = random.randint(3, 6)
                direction = random.choice([-1, 1])
                branch_x = current_x
                branch_y = y
                for i in range(branch_length):
                    branch_x += direction
                    branch_y -= 1
                    if random.random() < 0.8:
                        self.stem_pixels[(branch_x, branch_y)] = color
                    if i == branch_length - 1:
                        self._add_fixed_part(branch_x, branch_y, part_type='leaf')

        # 在主茎顶端添加叶子
        self._add_fixed_part(current_x, start_y - stem_height, part_type='leaf')

    def _add_branch(self, start_x, start_y, stem_height, direction):
        """添加分支，长度约为植株高度的1/3"""
        # 设置分支长度为植株高度的1/3左右
        target_length = int(stem_height * 4 / 9)
        base_length = target_length
        variation = 2    # 允许的变化范围
        branch_length = base_length + random.randint(-variation, variation)
        
        current_x = start_x
        current_y = start_y
        
        # 存储分支上的点
        branch_points = []
        
        # 确保分支的起始点连接在茎秆上
        self.stem_pixels[(current_x, current_y)] = random.choice(self.palettes['stem'])
        branch_points.append((current_x, current_y))
        
        # 创建分支，添加平缓的弯曲
        dx = direction * 0.8  # 降低初始倾斜度
        for i in range(branch_length):
            # 缓慢调整方向，使分支更平滑
            if random.random() < 0.15:  # 降低调整频率
                dx += random.choice([-0.2, 0, 0.2])  # 减小调整幅度
                dx = max(-1.2, min(1.2, dx))  # 限制最大倾斜度
            
            current_x += dx
            current_y -= 1  # 向上生长
            
            # 四舍五入到整数坐标
            pixel_x = round(current_x)
            
            # 检查位置是否已被占用
            if (pixel_x, current_y) not in self.stem_pixels:
                self.stem_pixels[(pixel_x, current_y)] = random.choice(self.palettes['stem'])
                branch_points.append((pixel_x, current_y))
        
        # 在分支生长完成后，随机添加1-3片叶子
        if branch_points:
            # 决定这个分支上要长几片叶子
            num_leaves = random.randint(1, 3)
            # 在分支上均匀分布叶子
            if num_leaves > 0:
                # 将分支点平均分配
                step = len(branch_points) // (num_leaves + 1)  # +1是为了避免叶子太密集
                for i in range(num_leaves):
                    point_index = (i + 1) * step  # 跳过分支起点
                    if point_index < len(branch_points):
                        point = branch_points[point_index]
                        self._add_fixed_part(point[0], point[1], 'leaf')
        return True

    def _grow_plant(self):
        """处理植物的生长"""
        if self.is_growing:
            return False
        self.is_growing = True

        stem_positions = list(self.stem_pixels.keys())
        if not stem_positions:
            self.is_growing = False
            return False

        # 获取当前茎干的最高点和最低点
        min_y = min(y for x, y in stem_positions)
        max_y = max(y for x, y in stem_positions)
        stem_height = max_y - min_y
        
        # 获取当前最高点的x坐标
        top_positions = [(x, y) for x, y in stem_positions if y == min_y]
        top_x = top_positions[0][0] if top_positions else self.grid_width // 2

        # 先尝试生长侧芽
        if self.force_branch or (min_y < self.grid_height * 0.3 and random.random() < 0.3):
            print("植物开始长出侧芽！🌿")
            
            # 简化侧芽生长点的选择逻辑
            possible_points = []
            last_growth_point = None
            
            # 从下往上遍历茎干位置
            sorted_positions = sorted(stem_positions, key=lambda pos: pos[1], reverse=True)
            
            for pos in sorted_positions:
                # 跳过最底部15%的区域
                if pos[1] > max_y - stem_height * 0.15:
                    continue
                    
                # 如果已经有一个生长点，检查与新点的距离
                if last_growth_point:
                    distance = abs(pos[1] - last_growth_point[1])
                    if distance < 10:  # 如果距离太近，跳过
                        continue
                
                # 检查左右是否有生长空间
                space_left = True
                space_right = True
                
                # 简化空间检查，只检查直接相邻的几个像素
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        check_pos = (pos[0] + dx, pos[1] + dy)
                        if check_pos in self.stem_pixels:
                            if dx < 0:
                                space_left = False
                            if dx > 0:
                                space_right = False
                
                # 如果左边或右边有足够空间
                if space_left or space_right:
                    possible_points.append((pos, space_left, space_right))
                    last_growth_point = pos
            
            # 如果找到了合适的生长点
            if possible_points:
                # 随机选择一个生长点
                growth_point, has_left_space, has_right_space = random.choice(possible_points)
                # 决定生长方向
                direction = random.choice([-1, 1]) if (has_left_space and has_right_space) else \
                          (-1 if has_left_space else 1)
                
                success = self._add_branch(growth_point[0], growth_point[1], stem_height, direction)
                if success:
                    self.force_branch = False
                    self.growth_stage += 1
                    self.ready_to_grow = False
                    self.is_growing = False
                    return True
            
            print("没有找到合适的侧芽生长点，植物将继续长高...")

        # 如果不能生长侧芽，则向上生长
        # 向上生长的距离（改为原来的1/4）
        growth_height = max(5, int(stem_height * 0.25))  # 确保至少有5个单位的生长高度
        
        # 从最高点开始生长
        current_x = float(top_x)
        start_y = min_y
        dx = 0
        
        # 检查当前生长方向趋势
        nearby_points = [(x, y) for x, y in stem_positions 
                       if abs(x - top_x) <= 3 and 
                          abs(y - min_y) <= 3]
        if len(nearby_points) >= 2:
            # 计算当前的生长趋势
            trend_dx = sum(p[0] - top_x for p in nearby_points) / len(nearby_points)
            # 稍微抵消当前趋势，避免过度倾斜
            dx = -trend_dx * 0.1
        
        # 生成新的茎干，从最高点开始
        new_stem_pixels = {}  # 临时存储新生成的茎秆像素
        for y in range(start_y - growth_height, start_y + 1):
            # 添加平滑的弯曲，但倾向于垂直生长
            if random.random() < 0.2:
                dx += random.choice([-0.2, 0, 0, 0.2])  # 增加保持直线的概率
                dx = max(-0.5, min(0.5, dx))  # 限制最大倾斜度
            current_x += dx
            pixel_x = round(current_x)
            
            # 更灵活地处理重叠问题
            overlap = False
            for offset in [-1, 0, 1]:
                if (pixel_x + offset, y) in self.stem_pixels:
                    overlap = True
                    break
            if overlap:
                # 尝试稍微偏移生长方向
                new_dx = random.choice([-0.2, 0.2])
                new_x = round(current_x + new_dx)
                if not any((new_x + offset, y) in self.stem_pixels for offset in [-1, 0, 1]):
                    pixel_x = new_x
                    overlap = False
            if not overlap:
                new_stem_pixels[(pixel_x, y)] = random.choice(self.palettes['stem'])
        
        # 确保新茎秆与原来的连接
        if new_stem_pixels:
            self.stem_pixels.update(new_stem_pixels)
            # 在新茎秆上添加1-2片叶子
            stem_points = list(new_stem_pixels.keys())
            if stem_points:
                num_leaves = random.randint(1, 2)
                for _ in range(num_leaves):
                    leaf_point = random.choice(stem_points)
                    self._add_fixed_part(leaf_point[0], leaf_point[1], 'leaf')
            print("植株长高了！🌱")
            self.growth_stage += 1
            self.ready_to_grow = False
            self.is_growing = False
            return True
        
        # 如果所有生长尝试都失败了，重置生长状态但不增加生长阶段
        print("植物暂时找不到合适的生长点，稍后再试...")
        self.ready_to_grow = False
        self.is_growing = False
        return False

    def add_part(self):
        """添加新的花朵或叶子（随机选择）"""
        # 如果植物已经准备好生长，强制处理生长
        if self.ready_to_grow:
            print("植物开始新的生长阶段！")
            return self._grow_plant()

        # 获取茎干的所有位置
        stem_positions = list(self.stem_pixels.keys())
        if stem_positions and len(self.parts) < self.max_parts:
            # 找出茎干的最高点和最低点
            min_y = min(y for x, y in stem_positions)  # 最高点（y值最小）
            max_y = max(y for x, y in stem_positions)  # 最低点（y值最大）
            stem_height = max_y - min_y

            # 计算可生长区域的高度范围
            safe_max_y = max_y - stem_height * 0.15  # 从底部往上15%开始（防止贴地生长）
            safe_min_y = min_y  # 上限是茎干顶部

            # 增加花朵生长的概率(70%)
            growth_type = random.choices(['flower', 'leaf'], weights=[70, 30])[0]
            valid_positions = []

            if growth_type == 'flower':
                # 简化花朵生长的位置选择，避免贴地生长
                for pos in stem_positions:
                    if safe_min_y <= pos[1] <= safe_max_y:
                        # 确保花朵不会贴地生长且不会长在新生成的茎秆上
                        if pos[1] < max_y - stem_height * 0.15:  # 至少在底部15%以上
                            # 检查是否是最近生成的茎秆
                            is_new_stem = False
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    if (pos[0] + dx, pos[1] + dy) in self.stem_pixels:
                                        # 如果这个茎秆是最近两次生长添加的，就不在这里长花
                                        if len(self.parts) - self.flower_count < 2:
                                            is_new_stem = True
                                            break
                            if not is_new_stem:
                                # 检查周围是否有足够的空间
                                has_space = True
                                for dx in [-3, -2, -1, 0, 1, 2, 3]:  # 扩大检查范围
                                    for dy in [-3, -2, -1, 0, 1, 2, 3]:
                                        if (pos[0] + dx, pos[1] + dy) in self.flower_pixels:
                                            has_space = False
                                            break
                                if has_space:
                                    valid_positions.append(pos)
            else:  # 叶子
                # 叶子在中上部生长
                for pos in stem_positions:
                    if safe_min_y <= pos[1] <= safe_max_y - stem_height * 0.3:
                        valid_positions.append(pos)

            # 如果没有找到合适的位置，使用所有非底部的茎干位置
            if not valid_positions:
                valid_positions = [pos for pos in stem_positions if pos[1] < safe_max_y]

            if valid_positions:
                # 选择生长位置
                max_attempts = 10
                for _ in range(max_attempts):
                    pos = random.choice(valid_positions)
                    x_offset = random.randint(-4, 4) if growth_type == 'flower' else random.randint(-2, 2)
                    y_offset = random.randint(-2, 2)
                    
                    success = self._add_fixed_part(
                        pos[0] + x_offset,
                        pos[1] + y_offset,
                        growth_type
                    )
                    
                    if success:
                        # 在新生长的位置添加粒子效果
                        growth_x = (pos[0] + x_offset) * self.pixel_size
                        growth_y = (pos[1] + y_offset) * self.pixel_size
                        self._add_particles(
                            growth_x,
                            growth_y,
                            self.growth_colors,
                            30
                        )
                        
                        # 添加浮动文字，位置居中
                        if growth_type == 'flower':
                            self._add_floating_text(
                                "A NEW FLOWER BLOOMS",  # 使用大写字母增强像素风格感
                                growth_x,  # 现在在_add_floating_text中会自动居中
                                growth_y - 30
                            )
                        else:
                            self._add_floating_text(
                                "A NEW LEAF GROWS",
                                growth_x,
                                growth_y - 30
                            )
                        
                        self.parts.append({'type': growth_type})
                        
                        if growth_type == 'flower':
                            self.flower_count += 1
                            print(f"植物长出了新的花朵！现在有 {self.flower_count} 朵花了 🌸")
                            if self.flower_count % 4 == 0:
                                self.ready_to_grow = True
                                self.force_branch = True
                                print("下一次点击将开始新的生长阶段，会长出侧芽！")
                            if self.flower_count % 3 == 0:
                                self._add_butterfly()
                                print("一只新的蝴蝶出现了！🦋")
                        else:
                            print("植物长出了新的叶子！🌱")
                        return True

        return False

    def remove_part(self):
        """移除一整片叶子或一整朵花"""
        if not self.flower_pixels and not self.leaf_pixels:
            return False

        # 找出所有花朵和叶子的中心点
        flower_centers = self._find_part_centers(self.flower_pixels)
        leaf_centers = self._find_part_centers(self.leaf_pixels)

        # 如果都没有找到中心点，返回False
        if not flower_centers and not leaf_centers:
            return False

        # 随机选择移除花朵或叶子
        if flower_centers and (not leaf_centers or random.random() < 0.5):
            # 移除一朵花
            center = random.choice(flower_centers)
            pixels_to_remove = self._get_connected_pixels(center, self.flower_pixels)
            
            # 在移除的位置添加粒子效果和文字
            center_x, center_y = center
            effect_x = center_x * self.pixel_size
            effect_y = center_y * self.pixel_size
            
            self._add_particles(
                effect_x,
                effect_y,
                self.wither_colors,
                40
            )
            
            self._add_floating_text(
                "A FLOWER WITHERS",
                effect_x,
                effect_y - 30
            )
            
            # 移除花朵的所有像素
            for pixel in pixels_to_remove:
                self.flower_pixels.pop(pixel, None)
        else:
            # 移除一片叶子
            center = random.choice(leaf_centers)
            pixels_to_remove = self._get_connected_pixels(center, self.leaf_pixels)
            
            # 在移除的位置添加粒子效果和文字
            center_x, center_y = center
            effect_x = center_x * self.pixel_size
            effect_y = center_y * self.pixel_size
            
            self._add_particles(
                effect_x,
                effect_y,
                self.wither_colors,
                30
            )
            
            self._add_floating_text(
                "A LEAF FALLS",
                effect_x,
                effect_y - 30
            )
            
            # 移除叶子的所有像素
            for pixel in pixels_to_remove:
                self.leaf_pixels.pop(pixel, None)

        return True

    def _find_part_centers(self, pixels_dict):
        """找出所有部件（花朵或叶子）的中心点"""
        if not pixels_dict:
            return []

        # 使用集合记录已经处理过的像素
        processed = set()
        centers = []

        for pixel in pixels_dict.keys():
            if pixel in processed:
                continue

            # 获取与当前像素相连的所有像素
            connected = self._get_connected_pixels(pixel, pixels_dict)
            processed.update(connected)

            # 计算这组像素的中心点
            if connected:
                avg_x = sum(x for x, y in connected) / len(connected)
                avg_y = sum(y for x, y in connected) / len(connected)
                # 找到最接近平均值的实际像素点
                center = min(connected, key=lambda p: ((p[0] - avg_x) ** 2 + (p[1] - avg_y) ** 2))
                centers.append(center)

        return centers

    def _get_connected_pixels(self, start_pixel, pixels_dict):
        """获取与给定像素相连的所有像素（使用广度优先搜索）"""
        connected = set()
        queue = [start_pixel]
        
        while queue:
            current = queue.pop(0)
            if current in connected:
                continue
                
            connected.add(current)
            x, y = current
            
            # 检查八个方向的相邻像素
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor = (x + dx, y + dy)
                    if neighbor in pixels_dict and neighbor not in connected:
                        queue.append(neighbor)
        
        return connected

    def _add_particles(self, x, y, colors, count):
        """在指定位置添加粒子效果"""
        for _ in range(count):
            # 随机偏移起始位置，使粒子效果更自然
            offset_x = random.randint(-5, 5)
            offset_y = random.randint(-5, 5)
            self.particles.append(Particle(x + offset_x, y + offset_y, colors))

    def draw(self):
        """绘制植物和背景"""
        # 检查文件是否有更新
        try:
            current_size = os.path.getsize(self.speech_file_path)
            if current_size != self.last_file_size:
                self.wordcloud_background = self._generate_wordcloud()
                self.last_file_size = current_size
        except OSError:
            # 如果文件不存在，创建一个黄色背景
            background = np.full((self.height, self.width, 3), (255, 223, 128), dtype=np.uint8)
            self.wordcloud_background = background
        
        # 创建基础图像
        base_image = self._create_pixel_grid()
        
        # 如果有词云背景，先绘制词云
        if self.wordcloud_background is not None:
            resized_wordcloud = cv2.resize(self.wordcloud_background, (self.width, self.height))
            wordcloud_pil = Image.fromarray(cv2.cvtColor(resized_wordcloud, cv2.COLOR_BGR2RGB))
            base_image = Image.alpha_composite(base_image.convert('RGBA'), wordcloud_pil.convert('RGBA'))
        else:
            # 如果没有词云，使用黄色背景
            background = np.full((self.height, self.width, 3), (255, 223, 128), dtype=np.uint8)
            background_pil = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
            base_image = Image.alpha_composite(base_image.convert('RGBA'), background_pil.convert('RGBA'))
        
        draw = ImageDraw.Draw(base_image)
        
        # 更新和绘制粒子
        self.particles = [p for p in self.particles if p.update()]
        for particle in self.particles:
            particle.draw(draw)
        
        # 更新和绘制浮动文字
        self.floating_texts = [text for text in self.floating_texts if text.update()]
        for text in self.floating_texts:
            text.draw(draw, self.font_path)
        
        # 绘制植物的其他部分
        self.update_butterflies()
        self._draw_pot(draw)
        self._draw_stem(draw)
        for (x, y), color in self.leaf_pixels.items():
            self._draw_pixel(draw, x, y, color)
        for (x, y), color in self.flower_pixels.items():
            self._draw_pixel(draw, x, y, color)
        
        # 绘制蝴蝶
        for b in self.butterflies:
            t = b['angle']
            x = int(b['center_x'] + b['a'] * math.sin(b['freq_x'] * t))
            y = int(b['center_y'] + b['b'] * math.sin(b['freq_y'] * t + b['phase']))
            self._draw_butterfly(draw, x, y, b['colors'], b['facing_right'])
        
        # 转换回OpenCV格式
        opencv_image = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
        return opencv_image

    def _resize_for_fullscreen(self, image):
        """调整图像大小以适应全屏显示，保持原有比例"""
        # 直接使用屏幕分辨率进行缩放
        resized = cv2.resize(image, (self.screen_width, self.screen_height), interpolation=cv2.INTER_NEAREST)
        return resized

    def update_butterflies(self):
        for b in self.butterflies:
            b['angle'] += b['speed']
            # 保证角度在合理范围内
            if b['angle'] > 100 * math.pi:  # 防止数值过大
                b['angle'] -= 100 * math.pi

    def _add_fixed_part(self, x, y, part_type=None, is_initial=False):
        """在指定位置添加固定的花朵或叶子"""
        if part_type is None:
            part_type = random.choices(['flower', 'leaf'], weights=[65, 35])[0]
            
        if part_type == 'flower':
            # 检查花朵重叠
            def calculate_overlap(test_x, test_y, pattern):
                overlap_pixels = 0
                total_pixels = len(pattern)
                for dx, dy in pattern:
                    test_pos = (test_x + dx, test_y + dy)
                    if test_pos in self.flower_pixels:
                        overlap_pixels += 1
                        # 如果重叠超过1/2，立即返回True
                        if overlap_pixels > total_pixels / 2:
                            return True
                return False
            
            # 选择花朵图案
            patterns = [
                # 大型花朵（菱形样式）
                [
                    # 中心部分
                    (0, 0), 
                    # 内层十字
                    (0, -1), (0, 1), (-1, 0), (1, 0),
                    # 内层对角
                    (-1, -1), (1, -1), (-1, 1), (1, 1),
                    # 外层十字
                    (0, -2), (0, 2), (-2, 0), (2, 0),
                    # 外层对角
                    (-2, -2), (2, -2), (-2, 2), (2, 2),
                    # 填充部分
                    (-1, -2), (1, -2), (-2, -1), (2, -1),
                    (-2, 1), (2, 1), (-1, 2), (1, 2),
                    # 最外层装饰
                    (0, -3), (0, 3), (-3, 0), (3, 0)
                ],
                # 另一种大型花朵（圆形样式）
                [
                    # 中心部分（黄色区域）
                    (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                    # 内层花瓣
                    (-2, -1), (-2, 0), (-2, 1),
                    (2, -1), (2, 0), (2, 1),
                    (-1, -2), (0, -2), (1, -2),
                    (-1, 2), (0, 2), (1, 2),
                    # 外层装饰
                    (-2, -2), (2, -2), (-2, 2), (2, 2),
                    (-3, 0), (3, 0), (0, -3), (0, 3)
                ]
            ]
            pattern = random.choice(patterns)
            
            # 检查重叠
            if calculate_overlap(x, y, pattern):
                # 如果重叠过多，触发生长
                self.ready_to_grow = True
                print("花朵太密集了，植物需要生长出新的空间...")
                return False
            
            # 如果没有过多重叠，正常添加花朵
            center_color = '#FFD700'  # 金黄色
            self.flower_pixels[(x, y)] = center_color
            
            # 绘制花瓣，使用渐变色
            petal_colors = self.selected_flower_palette
            for i, (dx, dy) in enumerate(pattern):
                # 跳过已经绘制的中心部分
                if (dx, dy) == (0, 0):
                    continue
                
                # 计算到中心的距离来决定使用哪种颜色
                distance = abs(dx) + abs(dy)
                color_idx = min(distance - 1, len(petal_colors) - 1)
                self.flower_pixels[(x + dx, y + dy)] = petal_colors[color_idx]
            
            return True
            
        else:  # leaf
            # 根据是否是初始叶子选择不同的图案
            if is_initial:
                # 初始叶子（更小）
                patterns = [
                    # 小型叶子
                    [
                        (0, 0), (1, 0),  # 叶子主体
                        (0, -1), (1, -1),  # 上部
                        (0, 1), (1, 1),    # 下部
                        (2, 0)  # 叶尖
                    ]
                ]
            else:
                # 后续生长的叶子（更大）
                patterns = [
                    # 中型叶子
                    [
                        (0, 0), (1, 0), (2, 0),  # 叶子主体
                        (0, -1), (1, -1), (2, -1),  # 上部
                        (0, 1), (1, 1), (2, 1),     # 下部
                        (3, 0), (3, -1), (3, 1)     # 叶尖加宽
                    ],
                    # 大型羽状叶
                    [
                        (0, 0), (1, 0), (2, 0), (3, 0),  # 主脉
                        (1, -1), (2, -1), (3, -1),      # 上部叶片
                        (1, 1), (2, 1), (3, 1),          # 下部叶片
                        (4, 0), (4, -1), (4, 1),         # 叶尖部分
                        (2, -2), (3, -2), (2, 2), (3, 2)  # 加宽的边缘
                    ]
                ]
            
            pattern = random.choice(patterns)
            
            # 检查是否与现有叶子重叠
            def check_overlap(test_x, test_y, pattern):
                for dx, dy in pattern:
                    test_pos = (test_x + dx, test_y + dy)
                    if test_pos in self.leaf_pixels:
                        return True
                return False
            
            # 尝试找到一个不重叠的位置
            max_attempts = 5
            original_x, original_y = x, y
            
            for _ in range(max_attempts):
                if not check_overlap(x, y, pattern):
                    break
                # 如果有重叠，稍微调整位置
                x = original_x + random.randint(-2, 2)
                y = original_y + random.randint(-2, 2)
            else:
                return False
            
            # 使用渐变色绘制叶子
            colors = self.palettes['leaf']
            for i, (dx, dy) in enumerate(pattern):
                # 根据到主脉的距离选择颜色
                distance = abs(dy)  # 使用垂直距离来决定颜色
                color_idx = min(distance, len(colors) - 1)
                self.leaf_pixels[(x + dx, y + dy)] = colors[color_idx]
            
            return True

    def _add_butterfly(self):
        center_x = random.randint(20, self.grid_width - 20)
        center_y = random.randint(20, self.grid_height - 20)
        # 修改参数以创建更复杂的轨迹
        a = random.uniform(10, 20)  # 轨迹范围
        b = random.uniform(10, 20)  # 轨迹范围
        freq_x = random.uniform(1, 3)  # x方向的频率
        freq_y = random.uniform(1, 3)  # y方向的频率
        phase = random.uniform(0, 2 * math.pi)  # 相位差
        speed = random.uniform(0.02, 0.05)  # 降低速度使运动更平滑

        # 生成随机颜色方案
        base_hue = random.random()
        # 生成协调的颜色方案
        light_color = self._hsv_to_hex(base_hue, 0.3, 1.0)  # 浅色
        medium_color = self._hsv_to_hex(base_hue, 0.6, 0.9)  # 中等色
        dark_color = self._hsv_to_hex(base_hue, 0.8, 0.7)  # 深色
        antenna_color = '#321900'  # 深棕色触角

        # 初始化时就决定蝴蝶的朝向
        facing_right = random.choice([True, False])

        self.butterflies.append({
            'center_x': center_x,
            'center_y': center_y,
            'a': a,
            'b': b,
            'freq_x': freq_x,
            'freq_y': freq_y,
            'phase': phase,
            'angle': 0,
            'speed': speed,
            'colors': {
                1: light_color,
                2: medium_color,
                3: dark_color,
                4: antenna_color
            },
            'facing_right': facing_right
        })

    def _hsv_to_hex(self, h, s, v):
        """将HSV颜色转换为十六进制颜色代码"""
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

    def _draw_butterfly(self, draw, x, y, colors, facing_right):
        """使用像素图案绘制蝴蝶"""
        # 蝴蝶像素图案（更小版本 5x7）
        butterfly_pattern = [
            [0,1,1,0,0], # 触角（改用浅色）
            [1,1,1,1,0], # 头部
            [1,1,2,1,0], # 上翅膀
            [1,2,2,2,1], # 上翅膀中部
            [1,3,3,2,0], # 下翅膀
            [0,3,3,2,0], # 身体
            [0,0,2,0,0], # 身体底部
        ]

        # 如果需要向左飞，翻转图案
        if not facing_right:
            butterfly_pattern = [row[::-1] for row in butterfly_pattern]

        # 绘制蝴蝶
        for row_idx, row in enumerate(butterfly_pattern):
            for col_idx, color_idx in enumerate(row):
                if color_idx == 0:  # 跳过透明部分
                    continue
                color = colors[color_idx]
                pixel_x = x + col_idx - len(butterfly_pattern[0]) // 2
                pixel_y = y + row_idx - len(butterfly_pattern) // 2
                self._draw_pixel(draw, pixel_x, pixel_y, color)

    def _create_pixel_grid(self):
        """创建像素网格"""
        return Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))  # 使用透明背景

    def _draw_pixel(self, draw, x, y, color):
        """绘制单个像素块"""
        x1 = x * self.pixel_size
        y1 = y * self.pixel_size
        x2 = x1 + self.pixel_size
        y2 = y1 + self.pixel_size
        draw.rectangle([x1, y1, x2, y2], fill=color)

    def _draw_pot(self, draw):
        """绘制已存储的花盆像素"""
        for (x, y), color in self.pot_pixels.items():
            self._draw_pixel(draw, x, y, color)

    def _draw_stem(self, draw):
        """绘制已存储的茎干像素"""
        for (x, y), color in self.stem_pixels.items():
            self._draw_pixel(draw, x, y, color)

    def start_listening(self):
        """启动语音识别线程"""
        self.speech_thread = threading.Thread(target=self._listen_for_speech)
        self.speech_thread.start()

    def stop_listening(self):
        """停止语音识别"""
        self.is_listening = False
        if self.speech_thread:
            self.speech_thread.join()

    def _listen_for_speech(self):
        """持续监听语音输入的线程函数"""
        print("开始监听语音...")
        
        # 配置识别器参数以提高准确性
        recognizer.energy_threshold = 300  # 声音识别阈值
        recognizer.dynamic_energy_threshold = True  # 动态调整阈值
        recognizer.pause_threshold = 1.2  # 增加停顿阈值，给说英语的人更多思考时间
        recognizer.phrase_threshold = 0.5  # 增加短语阈值
        recognizer.non_speaking_duration = 0.8  # 增加非说话判断时间
        
        with microphone as source:
            while self.is_listening:
                try:
                    print("\n正在调整噪音水平...")
                    # 增加环境噪音调整时间
                    recognizer.adjust_for_ambient_noise(source, duration=1.5)
                    
                    print("\n正在聆听...")
                    # 增加录音超时时间和短语时间限制
                    audio = recognizer.listen(source, timeout=7, phrase_time_limit=15)
                    
                    try:
                        # 首先尝试英文识别
                        text = recognizer.recognize_google(audio, language='en-US', show_all=True)
                        
                        # 处理英文识别结果
                        if isinstance(text, dict) and 'alternative' in text:
                            # 获取最可能的结果
                            text = text['alternative'][0]['transcript']
                            print(f"\n识别到的英文: {text}")
                            
                            # 使用ChatGLM分析情感
                            sentiment = analyze_sentiment_with_glm(text)
                            
                            if sentiment == 'positive':
                                with open(self.speech_file_path, "a", encoding="utf-8") as f:
                                    f.write(f"{text}\n")
                                print("检测到鼓励的话语！植物正在生长... 🌱")
                                
                                current_time = time.time()
                                if current_time - self.last_growth_time >= self.growth_cooldown:
                                    self.add_part()
                                    self.last_growth_time = current_time
                            elif sentiment == 'negative':
                                print("检测到消极的话语！植物有些难过... 🥀")
                                with open(self.speech_file_path, "a", encoding="utf-8") as f:
                                    f.write(f"{text} [negative]\n")
                                self.remove_part()  # 移除一部分植物
                            else:
                                print("继续说些鼓励的话语来帮助植物生长吧！")
                                
                        else:
                            # 如果英文识别失败，尝试中文识别
                            text = recognizer.recognize_google(audio, language='zh-CN')
                            
                            if text.strip():
                                print(f"\n识别到的中文: {text}")
                                
                                # 使用ChatGLM分析情感
                                sentiment = analyze_sentiment_with_glm(text)
                                
                                if sentiment == 'positive':
                                    with open(self.speech_file_path, "a", encoding="utf-8") as f:
                                        f.write(f"{text}\n")
                                    print("检测到鼓励的话语！植物正在生长... 🌱")
                                    
                                    current_time = time.time()
                                    if current_time - self.last_growth_time >= self.growth_cooldown:
                                        self.add_part()
                                        self.last_growth_time = current_time
                                elif sentiment == 'negative':
                                    print("检测到消极的话语！植物有些难过... 🥀")
                                    with open(self.speech_file_path, "a", encoding="utf-8") as f:
                                        f.write(f"{text} [negative]\n")
                                    self.remove_part()  # 移除一部分植物
                                else:
                                    print("继续说些鼓励的话语来帮助植物生长吧！")
                                    
                    except sr.UnknownValueError:
                        print("\n未能识别语音，请说得更清晰一些")
                    except sr.RequestError as e:
                        print(f"\n语音识别服务出错: {e}")
                        time.sleep(2)
                        
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"\n发生错误: {e}")
                    continue

    def _generate_wordcloud(self):
        """生成词云背景"""
        try:
            # 读取记录文件
            if not os.path.exists(self.speech_file_path):
                return None
                
            # 读取文件并过滤出积极的话语
            positive_texts = []
            with open(self.speech_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if '[negative]' not in line and '[neutral]' not in line:
                        # 移除可能存在的 [positive] 标记
                        text = line.replace('[positive]', '').strip()
                        positive_texts.append(text)
            
            if not positive_texts:
                return None
            
            # 将积极的话语合并
            text = ' '.join(positive_texts)
            
            if not text.strip():
                return None
            
            # 对中文文本进行分词
            words = ' '.join(jieba.cut(text))
            
            # 创建词云颜色函数 - 使用蓝色系渐变
            def word_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
                # 蓝色系渐变色组
                blue_colors = [
                    (135, 206, 250),  # 浅天蓝色
                    (100, 149, 237),  # 矢车菊蓝
                    (65, 105, 225),   # 皇家蓝
                    (30, 144, 255),   # 道奇蓝
                ]
                # 随机选择一个颜色
                color = random.choice(blue_colors)
                return f"rgb({color[0]}, {color[1]}, {color[2]})"
            
            # 创建词云
            wordcloud = WordCloud(
                width=self.width,
                height=self.height,
                background_color='white',  # 设置白色背景，后面会被天蓝色替换
                color_func=word_color_func,  # 使用自定义的蓝色系颜色函数
                font_path=self.font_path,  # 使用找到的字体
                prefer_horizontal=0.7,  # 70%的词水平显示
                min_font_size=8,  # 最小字体大小
                max_font_size=80,  # 最大字体大小
                relative_scaling=0.5,  # 调整词频和字体大小的关系
                font_step=1,  # 字体大小的步进值，设置小一点让字体大小更丰富
                random_state=42,  # 固定随机状态，使布局更稳定
                mask=None,  # 不使用遮罩
                mode='RGBA'  # 使用RGBA模式支持透明度
            ).generate(words)
            
            # 转换为带透明度的图像
            wordcloud_image = wordcloud.to_array()
            
            # 创建天蓝色背景
            background = np.full((self.height, self.width, 3), (135, 206, 235), dtype=np.uint8)  # 天蓝色 (135, 206, 235)
            
            # 将词云转换为PIL图像以处理透明度
            wordcloud_pil = Image.fromarray(wordcloud_image)
            
            # 调整词云的透明度
            wordcloud_pil.putalpha(128)  # 设置50%的透明度
            
            # 将PIL图像转换回numpy数组
            wordcloud_array = np.array(wordcloud_pil)
            
            # 将词云叠加到天蓝色背景上
            background_pil = Image.fromarray(background)
            background_pil.paste(wordcloud_pil, (0, 0), wordcloud_pil)
            
            return cv2.cvtColor(np.array(background_pil), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"生成词云时出错: {e}")
            return None

    def _get_font_path(self):
        """获取可用的字体路径"""
        font_paths = [
            'C:/Windows/Fonts/NotoSansSC-Black.ttf',     # Noto Sans SC Black
            'C:/Windows/Fonts/NotoSansSC-Bold.ttf',      # Noto Sans SC Bold
            'C:/Windows/Fonts/msyh.ttc',                 # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf'                # 黑体
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                return path
        return None

    def _add_floating_text(self, text, x, y):
        """添加浮动文字"""
        if self.font_path:
            # 使用黑色半透明文字
            self.floating_texts.append(FloatingText(text, x, y, '#000000'))

class PixelPlantApp:
    def __init__(self):
        self.is_running = True
        self.plant = PixelPlant()
        self.window_name = 'Pixel Plant'
        self.is_fullscreen = False
        self.screen_width = 3072
        self.screen_height = 1920

    def run(self):
        """运行应用程序"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # 设置初始窗口大小
        initial_scale = 2
        cv2.resizeWindow(self.window_name,
                        self.plant.base_width * initial_scale,
                        self.plant.base_height * initial_scale)

        print("🌱 像素盆栽已准备就绪！")
        print("说些鼓励的话让植物生长")
        print("按 'f' 切换全屏")
        print("按 'q' 退出程序")

        # 启动语音识别
        self.plant.start_listening()

        while True:
            frame = self.plant.draw()

            if self.is_fullscreen:
                frame = self._resize_for_fullscreen(frame)

            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
                break
            elif key == ord('f'):
                self.is_fullscreen = not self.is_fullscreen
                if self.is_fullscreen:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.window_name,
                                   self.plant.base_width * initial_scale,
                                   self.plant.base_height * initial_scale)

        # 清理资源
        self.plant.stop_listening()
        cv2.destroyAllWindows()
        engine.stop()

    def _resize_for_fullscreen(self, image):
        """调整图像大小以适应全屏显示，保持原有比例"""
        resized = cv2.resize(image, (self.screen_width, self.screen_height), interpolation=cv2.INTER_NEAREST)
        return resized


if __name__ == "__main__":
    app = PixelPlantApp()
    app.run()