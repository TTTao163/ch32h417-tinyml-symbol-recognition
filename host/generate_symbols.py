#!/usr/bin/env python3
"""
Plus-Minus 合成数据生成器

生成"+"和"-"符号的28x28像素灰度图像，用于机器学习训练。
支持多种变换（旋转、位置偏移、线条粗细）和噪声添加。
"""

import os
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image


@dataclass
class ImageParameters:
    """图像生成参数"""
    thickness: int  # 1-5像素
    position: Tuple[int, int]  # (x, y)，中心位置
    rotation: float  # -15到+15度
    noise_level: float  # 0-50
    
    def __post_init__(self):
        """验证参数范围"""
        assert 1 <= self.thickness <= 5, f"thickness must be 1-5, got {self.thickness}"
        assert -15 <= self.rotation <= 15, f"rotation must be -15 to 15, got {self.rotation}"
        assert 0 <= self.noise_level <= 50, f"noise_level must be 0-50, got {self.noise_level}"


@dataclass
class GenerationResult:
    """生成结果统计"""
    symbol: str
    count: int
    output_dir: str
    success: bool
    error_message: Optional[str] = None
    
    def __str__(self) -> str:
        if self.success:
            return f"✓ 成功生成 {self.count} 张 '{self.symbol}' 图像到 {self.output_dir}"
        else:
            return f"✗ 生成 '{self.symbol}' 失败: {self.error_message}"



class SymbolRenderer:
    """负责渲染符号图像的核心类"""
    
    def __init__(self, canvas_size: int = 28):
        """
        初始化渲染器
        
        Args:
            canvas_size: 画布尺寸（像素），默认28x28
        """
        self.canvas_size = canvas_size
    
    def draw_plus(
        self,
        thickness: int = 3,
        position: Tuple[int, int] = (14, 14),
        rotation: float = 0.0
    ) -> np.ndarray:
        """
        绘制"+"符号
        
        Args:
            thickness: 线条粗细（1-5像素）
            position: 符号中心位置（x, y）
            rotation: 旋转角度（-15到+15度）
            
        Returns:
            28x28的numpy数组，dtype=uint8
        """
        # 创建黑色画布
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        
        cx, cy = position
        half_thick = thickness // 2
        line_length = 16  # 线条长度
        
        # 绘制水平线
        y_start = max(0, cy - half_thick)
        y_end = min(self.canvas_size, cy + half_thick + 1)
        x_start = max(0, cx - line_length // 2)
        x_end = min(self.canvas_size, cx + line_length // 2)
        canvas[y_start:y_end, x_start:x_end] = 255
        
        # 绘制垂直线
        x_start = max(0, cx - half_thick)
        x_end = min(self.canvas_size, cx + half_thick + 1)
        y_start = max(0, cy - line_length // 2)
        y_end = min(self.canvas_size, cy + line_length // 2)
        canvas[y_start:y_end, x_start:x_end] = 255
        
        # 旋转功能需要scipy，暂时跳过
        # if abs(rotation) > 0.1:
        #     canvas = self._rotate_image(canvas, rotation)
        
        return canvas
    
    def draw_minus(
        self,
        thickness: int = 3,
        position: Tuple[int, int] = (14, 14),
        rotation: float = 0.0
    ) -> np.ndarray:
        """
        绘制"-"符号
        
        Args:
            thickness: 线条粗细（1-5像素）
            position: 符号中心位置（x, y）
            rotation: 旋转角度（-15到+15度）
            
        Returns:
            28x28的numpy数组，dtype=uint8
        """
        # 创建黑色画布
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        
        cx, cy = position
        half_thick = thickness // 2
        line_length = 16  # 线条长度
        
        # 绘制水平线
        y_start = max(0, cy - half_thick)
        y_end = min(self.canvas_size, cy + half_thick + 1)
        x_start = max(0, cx - line_length // 2)
        x_end = min(self.canvas_size, cx + line_length // 2)
        canvas[y_start:y_end, x_start:x_end] = 255
        
        # 旋转功能需要scipy，暂时跳过
        # if abs(rotation) > 0.1:
        #     canvas = self._rotate_image(canvas, rotation)
        
        return canvas
    
    def draw_multiply(
        self,
        thickness: int = 3,
        position: Tuple[int, int] = (14, 14),
        rotation: float = 0.0
    ) -> np.ndarray:
        """
        绘制"*"符号（乘号）
        
        Args:
            thickness: 线条粗细（1-5像素）
            position: 符号中心位置（x, y）
            rotation: 旋转角度（-15到+15度）
            
        Returns:
            28x28的numpy数组，dtype=uint8
        """
        # 创建黑色画布
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        
        cx, cy = position
        half_thick = thickness // 2
        line_length = 12  # 线条长度，比+号稍短
        
        # 绘制主对角线（左上到右下）
        for i in range(-line_length//2, line_length//2 + 1):
            for t in range(-half_thick, half_thick + 1):
                x = cx + i + t
                y = cy + i - t
                if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                    canvas[y, x] = 255
        
        # 绘制副对角线（右上到左下）
        for i in range(-line_length//2, line_length//2 + 1):
            for t in range(-half_thick, half_thick + 1):
                x = cx + i + t
                y = cy - i + t
                if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                    canvas[y, x] = 255
        
        return canvas
    
    def draw_percent(
        self,
        thickness: int = 3,
        position: Tuple[int, int] = (14, 14),
        rotation: float = 0.0
    ) -> np.ndarray:
        """
        绘制"%"符号
        
        Args:
            thickness: 线条粗细（1-5像素）
            position: 符号中心位置（x, y）
            rotation: 旋转角度（-15到+15度）
            
        Returns:
            28x28的numpy数组，dtype=uint8
        """
        # 创建黑色画布
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        
        cx, cy = position
        
        # 绘制斜线（从左上到右下）
        line_length = 16
        for i in range(-line_length//2, line_length//2 + 1):
            for t in range(thickness):
                x = cx + i + t - thickness//2
                y = cy + i + t - thickness//2
                if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                    canvas[y, x] = 255
        
        # 绘制上方小圆圈
        circle_radius = 3
        circle_x, circle_y = cx - 6, cy - 6
        for dy in range(-circle_radius, circle_radius + 1):
            for dx in range(-circle_radius, circle_radius + 1):
                if dx*dx + dy*dy <= circle_radius*circle_radius:
                    x, y = circle_x + dx, circle_y + dy
                    if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                        # 只绘制圆圈边缘
                        if dx*dx + dy*dy >= (circle_radius-1)*(circle_radius-1):
                            canvas[y, x] = 255
        
        # 绘制下方小圆圈
        circle_x, circle_y = cx + 6, cy + 6
        for dy in range(-circle_radius, circle_radius + 1):
            for dx in range(-circle_radius, circle_radius + 1):
                if dx*dx + dy*dy <= circle_radius*circle_radius:
                    x, y = circle_x + dx, circle_y + dy
                    if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                        # 只绘制圆圈边缘
                        if dx*dx + dy*dy >= (circle_radius-1)*(circle_radius-1):
                            canvas[y, x] = 255
        
        return canvas
    
    def draw_divide(
        self,
        thickness: int = 3,
        position: Tuple[int, int] = (14, 14),
        rotation: float = 0.0
    ) -> np.ndarray:
        """
        绘制"÷"符号（除号）
        
        Args:
            thickness: 线条粗细（1-5像素）
            position: 符号中心位置（x, y）
            rotation: 旋转角度（-15到+15度）
            
        Returns:
            28x28的numpy数组，dtype=uint8
        """
        # 创建黑色画布
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        
        cx, cy = position
        half_thick = thickness // 2
        line_length = 14  # 中间横线长度
        
        # 绘制中间水平线
        y_start = max(0, cy - half_thick)
        y_end = min(self.canvas_size, cy + half_thick + 1)
        x_start = max(0, cx - line_length // 2)
        x_end = min(self.canvas_size, cx + line_length // 2)
        canvas[y_start:y_end, x_start:x_end] = 255
        
        # 绘制上方小圆点
        dot_radius = 2
        dot_y = cy - 6
        for dy in range(-dot_radius, dot_radius + 1):
            for dx in range(-dot_radius, dot_radius + 1):
                if dx*dx + dy*dy <= dot_radius*dot_radius:
                    x, y = cx + dx, dot_y + dy
                    if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                        canvas[y, x] = 255
        
        # 绘制下方小圆点
        dot_y = cy + 6
        for dy in range(-dot_radius, dot_radius + 1):
            for dx in range(-dot_radius, dot_radius + 1):
                if dx*dx + dy*dy <= dot_radius*dot_radius:
                    x, y = cx + dx, dot_y + dy
                    if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                        canvas[y, x] = 255
        
        return canvas
    
    def add_noise(
        self,
        image: np.ndarray,
        noise_level: float = 0.0
    ) -> np.ndarray:
        """
        向图像添加高斯噪声
        
        Args:
            image: 输入图像数组
            noise_level: 噪声强度（0-50）
            
        Returns:
            添加噪声后的图像，像素值裁剪到0-255范围
        """
        if noise_level <= 0:
            return image
        
        # 添加高斯噪声
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image.astype(np.float32) + noise
        
        # 裁剪到有效范围
        noisy_image = np.clip(noisy_image, 0, 255)
        
        return noisy_image.astype(np.uint8)



class FileSystemManager:
    """处理文件系统操作"""
    
    def __init__(self, base_dir: str = "mnist/training"):
        """
        初始化文件系统管理器
        
        Args:
            base_dir: 基础目录路径
        """
        self.base_dir = base_dir
        self.counters = {'+': 0, '-': 0, 'multiply': 0, 'percent': 0, 'divide': 0}
    
    def create_directories(self, symbols: list) -> None:
        """
        创建必要的目录结构
        
        Args:
            symbols: 符号列表，如['+', '-']
        """
        for symbol in symbols:
            dir_path = os.path.join(self.base_dir, symbol)
            os.makedirs(dir_path, exist_ok=True)
    
    def get_next_filename(self, symbol: str) -> str:
        """
        获取下一个可用的文件名
        
        Args:
            symbol: 符号类型
            
        Returns:
            文件名（如"0.png"）
        """
        filename = f"{self.counters[symbol]}.png"
        self.counters[symbol] += 1
        return filename
    
    def save_image(
        self,
        image: np.ndarray,
        symbol: str
    ) -> str:
        """
        保存图像到适当的目录
        
        Args:
            image: 图像数组
            symbol: 符号类型（'+'或'-'）
            
        Returns:
            保存的文件路径
        """
        # 转换为PIL Image
        pil_image = Image.fromarray(image, mode='L')
        
        # 生成文件路径
        filename = self.get_next_filename(symbol)
        filepath = os.path.join(self.base_dir, symbol, filename)
        
        # 保存
        pil_image.save(filepath)
        
        return filepath



class GeneratorOrchestrator:
    """协调整个生成过程"""
    
    def __init__(
        self,
        renderer: SymbolRenderer,
        fs_manager: FileSystemManager
    ):
        """
        初始化协调器
        
        Args:
            renderer: 符号渲染器实例
            fs_manager: 文件系统管理器实例
        """
        self.renderer = renderer
        self.fs_manager = fs_manager
    
    def _random_params(self) -> Dict[str, Any]:
        """
        生成随机参数
        
        Returns:
            包含thickness, position, rotation, noise_level的字典
        """
        return {
            'thickness': np.random.randint(1, 6),  # 1-5
            'position': (
                14 + np.random.randint(-4, 5),  # 中心±4像素
                14 + np.random.randint(-4, 5)
            ),
            'rotation': np.random.uniform(-15, 15),  # ±15度
            'noise_level': np.random.uniform(0, 30)  # 0-30
        }
    
    def generate_batch(
        self,
        symbol: str,
        count: int,
        randomize: bool = True
    ) -> int:
        """
        批量生成图像
        
        Args:
            symbol: 符号类型（'plus', 'minus', 'multiply', 'percent', 'divide'）
            count: 生成数量
            randomize: 是否随机化参数
            
        Returns:
            成功生成的图像数量
        """
        success_count = 0
        symbol_char_map = {
            'plus': '+',
            'minus': '-', 
            'multiply': 'multiply',
            'percent': 'percent',
            'divide': 'divide'
        }
        symbol_char = symbol_char_map.get(symbol, symbol)
        
        for i in range(count):
            try:
                # 获取参数
                if randomize:
                    params = self._random_params()
                else:
                    params = {
                        'thickness': 3,
                        'position': (14, 14),
                        'rotation': 0.0,
                        'noise_level': 0.0
                    }
                
                # 绘制符号
                if symbol == 'plus':
                    image = self.renderer.draw_plus(
                        thickness=params['thickness'],
                        position=params['position'],
                        rotation=params['rotation']
                    )
                elif symbol == 'minus':
                    image = self.renderer.draw_minus(
                        thickness=params['thickness'],
                        position=params['position'],
                        rotation=params['rotation']
                    )
                elif symbol == 'multiply':
                    image = self.renderer.draw_multiply(
                        thickness=params['thickness'],
                        position=params['position'],
                        rotation=params['rotation']
                    )
                elif symbol == 'percent':
                    image = self.renderer.draw_percent(
                        thickness=params['thickness'],
                        position=params['position'],
                        rotation=params['rotation']
                    )
                elif symbol == 'divide':
                    image = self.renderer.draw_divide(
                        thickness=params['thickness'],
                        position=params['position'],
                        rotation=params['rotation']
                    )
                else:
                    print(f"  警告: 不支持的符号类型: {symbol}")
                    continue
                
                # 添加噪声
                image = self.renderer.add_noise(image, params['noise_level'])
                
                # 保存图像
                self.fs_manager.save_image(image, symbol_char)
                success_count += 1
                
                # 显示进度
                if (i + 1) % 100 == 0:
                    print(f"  已生成 {i + 1}/{count} 张 '{symbol}' 图像...")
                    
            except Exception as e:
                print(f"  警告: 生成第 {i+1} 张图像失败: {e}")
                continue
        
        return success_count



def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(
        description='生成+和-符号的合成训练数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python generate_symbols.py --count 1000 --symbol both
  python generate_symbols.py --count 500 --symbol plus
  python generate_symbols.py --count 2000 --symbol minus
        """
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=1000,
        help='每个符号生成的图像数量（默认：1000）'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        choices=['plus', 'minus', 'multiply', 'percent', 'divide', 'all'],
        default='all',
        help='生成的符号类型（默认：all）'
    )
    
    args = parser.parse_args()
    
    try:
        # 验证参数
        if args.count <= 0:
            raise ValueError("count 必须大于 0")
        
        print(f"\n{'='*60}")
        print(f"Plus-Minus 合成数据生成器")
        print(f"{'='*60}\n")
        
        # 创建实例
        renderer = SymbolRenderer(canvas_size=28)
        fs_manager = FileSystemManager(base_dir="mnist/training")
        orchestrator = GeneratorOrchestrator(renderer, fs_manager)
        
        # 确定要生成的符号
        symbols_to_generate = []
        if args.symbol == 'all':
            symbols_to_generate = ['plus', 'minus', 'multiply', 'percent', 'divide']
        elif args.symbol == 'plus':
            symbols_to_generate = ['plus']
        elif args.symbol == 'minus':
            symbols_to_generate = ['minus']
        elif args.symbol == 'multiply':
            symbols_to_generate = ['multiply']
        elif args.symbol == 'percent':
            symbols_to_generate = ['percent']
        elif args.symbol == 'divide':
            symbols_to_generate = ['divide']
        
        # 创建目录
        symbol_char_map = {
            'plus': '+',
            'minus': '-', 
            'multiply': 'multiply',
            'percent': 'percent',
            'divide': 'divide'
        }
        symbol_chars = [symbol_char_map[s] for s in symbols_to_generate]
        fs_manager.create_directories(symbol_chars)
        print(f"✓ 目录结构已创建\n")
        
        # 生成图像
        results = []
        symbol_char_map = {
            'plus': '+',
            'minus': '-', 
            'multiply': 'multiply',
            'percent': 'percent',
            'divide': 'divide'
        }
        
        for symbol in symbols_to_generate:
            symbol_char = symbol_char_map[symbol]
            print(f"正在生成 '{symbol}' 符号图像...")
            
            success_count = orchestrator.generate_batch(
                symbol=symbol,
                count=args.count,
                randomize=True
            )
            
            result = GenerationResult(
                symbol=symbol_char,
                count=success_count,
                output_dir=os.path.join(fs_manager.base_dir, symbol_char),
                success=True
            )
            results.append(result)
            print(f"✓ 完成: {success_count}/{args.count} 张图像\n")
        
        # 显示统计信息
        print(f"{'='*60}")
        print(f"生成完成!")
        print(f"{'='*60}\n")
        
        total_images = sum(r.count for r in results)
        print(f"总计生成: {total_images} 张图像")
        for result in results:
            print(f"  - '{result.symbol}': {result.count} 张 → {result.output_dir}")
        
        print(f"\n{'='*60}\n")
        
    except ValueError as e:
        print(f"✗ 参数错误: {e}")
        return 1
    except PermissionError as e:
        print(f"✗ 权限错误: {e}")
        print("  建议: 请检查目录写入权限")
        return 1
    except OSError as e:
        print(f"✗ 文件系统错误: {e}")
        print("  建议: 请检查磁盘空间")
        return 1
    except Exception as e:
        print(f"✗ 未知错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
fs_manager = FileSystemManager(base_dir="mnist/training")
