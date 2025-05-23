from skimage import io, data
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def RLE_encode_with_single_channel(img):
    code = []
    flat = img.flatten()
    prev_pixel = flat[0]
    count = 1

    for pixel in flat[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            code.append((count, prev_pixel))
            prev_pixel = pixel
            count = 1
    code.append((count, prev_pixel))

    return code
    

def RLE_encode(img):
    code = {}
    shape = img.shape

    for i in range(shape[2]):
        code[i] = RLE_encode_with_single_channel(img[:, :, i])

    return code, shape

def RLE_decode_with_single_channel(code, shape):
    decoded_img = np.zeros(shape, dtype=int)
    index = 0

    for count, pixel in code:
        for _ in range(count):
            decoded_img[index // shape[1]][index % shape[1]] = pixel
            index += 1

    return decoded_img

def RLE_decode(code, shape):
    decoded_img = np.zeros(shape, dtype=int)

    for i in range(shape[2]):
        decoded_img[:, :, i] = RLE_decode_with_single_channel(code[i], shape[:2])

    return decoded_img

def RLE_compress_rate(code, shape):
    original_size = shape[0] * shape[1] * shape[2] * 8
    compressed_size = 0

    for i in range(shape[2]):
        for count, pixel in code[i]:
            compressed_size += (len(bin(count)) - 2) + (len(bin(pixel)) - 2)

    return compressed_size / original_size

class TreeNode():
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.code = None
        self.code_length = 0
        self.frequency = 0

def code_assign(node, code):
    if node.left is None and node.right is None:
        node.code = code
        node.code_length = len(code)
    else:
        if node.left:
            code_assign(node.left, code + '0')
        if node.right:
            code_assign(node.right, code + '1')

def get_code(node, code={}):
    if node.left is None and node.right is None:
        code[node.value] = node.code
    else:
        if node.left:
            get_code(node.left, code)
        if node.right:
            get_code(node.right, code)

def get_code_length(node, code_length={}):
    if node.left is None and node.right is None:
        code_length[node.value] = node.code_length
    else:
        if node.left:
            get_code_length(node.left, code_length)
        if node.right:
            get_code_length(node.right, code_length)

def build_huffman_tree(img):
    freq = {}
    for row in img:
        for pixel in row:
            if pixel in freq:
                freq[pixel] += 1
            else:
                freq[pixel] = 1

    nodes = [TreeNode(value) for value in freq.keys()]
    for i, node in enumerate(nodes):
        node.frequency = list(freq.values())[i]

    while len(nodes) > 1:
        nodes.sort(key=lambda x: x.frequency)
        left = nodes.pop(0)
        right = nodes.pop(0)

        new_node = TreeNode(None)
        new_node.left = left
        new_node.right = right
        new_node.frequency = left.frequency + right.frequency

        nodes.append(new_node)

    root = nodes[0]
    code_assign(root, '')
    
    code = {}
    get_code(root, code)

    code_length = {}
    get_code_length(root, code_length)

    return root, code, code_length

def huffman_encode(img):
    shape = img.shape
    code_map = {}
    code_length_map = {}
    encoded_img = {}

    for i in range(shape[2]):
        root, code, code_length = build_huffman_tree(img[:, :, i])
        encoded_channel = []
        for row in img[:, :, i]:
            for pixel in row:
                encoded_channel.append(code[pixel])
        encoded_img[i] = encoded_channel
        code_map[i] = code
        code_length_map[i] = code_length

    return encoded_img, code_map, code_length_map, shape

def huffman_decode(encoded_img, code_map, shape):
    decoded_img = np.zeros(shape, dtype=int)

    for i in range(shape[2]):
        channel_encoded = encoded_img[i]
        code = code_map[i]
        index = 0
        for bits in channel_encoded:
            for value, huffman_code in code.items():
                if bits == huffman_code:
                    decoded_img[index // shape[1], index % shape[1], i] = value
                    index += 1
                    break

    return decoded_img

def Huffman_compress_rate(encoded_img, code_map, shape):
    original_size = shape[0] * shape[1] * shape[2] * 8
    compressed_size = 0

    for i in range(shape[2]):
        for bits in encoded_img[i]:
            compressed_size += len(bits)

    return  compressed_size / original_size

def dct_1d(img, nc):
    n = len(img)
    new_img = np.zeros_like(img, dtype=float)
    for k in range(n):
        sum_val = 0.0
        for i in range(n):
            sum_val += img[i] * np.cos(np.pi * k * (2 * i + 1) / (2 * n))
        if k == 0:
            new_img[k] = sum_val * np.sqrt(1 / n)
        else:
            new_img[k] = sum_val * np.sqrt(2 / n)
    
    nc_clamped = min(nc, n)
    if nc_clamped < n:
        new_img[nc_clamped:] = 0
    
    return new_img

def idct_1d(img):
    n = len(img)
    new_img = np.zeros_like(img, dtype=float)
    for i in range(n):
        sum_val = 0.0
        for k in range(n):
            term = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
            if k == 0:
                sum_val += img[k] * np.sqrt(1 / n) * term
            else:
                sum_val += img[k] * np.sqrt(2 / n) * term
        new_img[i] = sum_val
    
    return new_img

def dct_2d_block(img_block, nc):
    temp = np.zeros_like(img_block, dtype=float)
    result = np.zeros_like(img_block, dtype=float)
    for i in range(img_block.shape[0]):
        temp[i, :] = dct_1d(img_block[i, :], nc)
    for j in range(img_block.shape[1]):
        result[:, j] = dct_1d(temp[:, j], nc)
    return result

def idct_2d_block(img_block):
    temp = np.zeros_like(img_block, dtype=float)
    result = np.zeros_like(img_block, dtype=float)
    for i in range(img_block.shape[0]):
        temp[i, :] = idct_1d(img_block[i, :])
    for j in range(img_block.shape[1]):
        result[:, j] = idct_1d(temp[:, j])
    return result

def dct_2d_multichannel(img, nc, block_size=8):
    height, width, channels = img.shape
    dct_img = np.zeros_like(img, dtype=float)
    for c in range(channels):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                end_i = min(i + block_size, height)
                end_j = min(j + block_size, width)
                block = img[i:end_i, j:end_j, c]
                dct_block = dct_2d_block(block, nc)
                dct_img[i:end_i, j:end_j, c] = dct_block
    return dct_img

def idct_2d_multichannel(dct_img, block_size=8):
    height, width, channels = dct_img.shape
    idct_img = np.zeros_like(dct_img, dtype=float)
    for c in range(channels):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                end_i = min(i + block_size, height)
                end_j = min(j + block_size, width)
                block = dct_img[i:end_i, j:end_j, c]
                idct_block = idct_2d_block(block)
                idct_img[i:end_i, j:end_j, c] = idct_block
    return idct_img

def dct_compress_rate(img, nc, block_size=8):
    height, width, channels = img.shape
    original_size = height * width * channels * 8
    compressed_size = 0

    for c in range(channels):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                end_i = min(i + block_size, height)
                end_j = min(j + block_size, width)
                block = img[i:end_i, j:end_j, c]
                dct_block = dct_2d_block(block, nc)
                compressed_size += np.count_nonzero(dct_block)

    return compressed_size / original_size

import numpy as np
import os
from PIL import Image

class KJPEG:
    def __init__(self):
        # 初始化DCT变换的A矩阵
        self.__dctA = np.zeros(shape=(8, 8))
        for i in range(8):
            c = 0
            if i == 0:
                c = np.sqrt(1 / 8)
            else:
                c = np.sqrt(2 / 8)
            for j in range(8):
                self.__dctA[i, j] = c * np.cos(np.pi * i * (2 * j + 1) / (2 * 8))
        # 亮度量化矩阵
        self.__lq = np.array([
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99,
        ])
        # 色度量化矩阵
        self.__cq = np.array([
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
        ])
        # 标记矩阵类型，lt是亮度矩阵，ct是色度矩阵
        self.__lt = 0
        self.__ct = 1
        # Zig编码表
        self.__zig = np.array([
            0, 1, 8, 16, 9, 2, 3, 10,
            17, 24, 32, 25, 18, 11, 4, 5,
            12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13, 6, 7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36,
            29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46,
            53, 60, 61, 54, 47, 55, 62, 63
        ])
        # Zag编码表
        self.__zag = np.array([
            0, 1, 5, 6, 14, 15, 27, 28,
            2, 4, 7, 13, 16, 26, 29, 42,
            3, 8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 41, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
        ])

    def __Rgb2Yuv(self, r, g, b):
        # 从图像获取YUV矩阵
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
        v = 0.5 * r - 0.419 * g - 0.081 * b + 128
        return y, u, v

    def __Fill(self, matrix):
        # 图片的长宽都需要满足是16的倍数（采样长宽会缩小1/2和取块长宽会缩小1/8）
        # 图像压缩三种取样方式4:4:4、4:2:2、4:2:0
        fh, fw = 0, 0
        if self.height % 16 != 0:
            fh = 16 - self.height % 16
        if self.width % 16 != 0:
            fw = 16 - self.width % 16
        res = np.pad(matrix, ((0, fh), (0, fw)), 'constant',
                             constant_values=(0, 0))
        return res

    def __Encode(self, matrix, tag):
        # 先对矩阵进行填充
        matrix = self.__Fill(matrix)
        # 将图像矩阵切割成8*8小块
        height, width = matrix.shape
        # 减少for循环语句，利用numpy的自带函数来提升算法效率
        # 参考吴恩达的公开课视频，numpy的函数自带并行处理，不用像for循环一样串行处理
        shape = (height // 8, width // 8, 8, 8)
        strides = matrix.itemsize * np.array([width * 8, 8, width, 1])
        blocks = np.lib.stride_tricks.as_strided(matrix, shape=shape, strides=strides)
        res = []
        for i in range(height // 8):
            for j in range(width // 8):
                res.append(self.__Quantize(self.__Dct(blocks[i, j]).reshape(64), tag))
        return res

    def __Dct(self, block):
        # DCT变换
        res = np.dot(self.__dctA, block)
        res = np.dot(res, np.transpose(self.__dctA))
        return res

    def __Quantize(self, block, tag):
        res = block
        if tag == self.__lt:
            res = np.round(res / self.__lq)
        elif tag == self.__ct:
            res = np.round(res / self.__cq)
        return res

    def __Zig(self, blocks):
        ty = np.array(blocks)
        tz = np.zeros(ty.shape)
        for i in range(len(self.__zig)):
            tz[:, i] = ty[:, self.__zig[i]]
        tz = tz.reshape(tz.shape[0] * tz.shape[1])
        return tz.tolist()

    def __Rle(self, blist):
        res = []
        cnt = 0
        for i in range(len(blist)):
            if blist[i] != 0:
                res.append(cnt)
                res.append(int(blist[i]))
                cnt = 0
            elif cnt == 15:
                res.append(cnt)
                res.append(int(blist[i]))
                cnt = 0
            else:
                cnt += 1
        # 末尾全是0的情况
        if cnt != 0:
            res.append(cnt - 1)
            res.append(0)
        return res

    def Compress(self, filename):
        # 根据路径image_path读取图片，并存储为RGB矩阵
        image = Image.open(filename)
        # 获取图片宽度width和高度height
        self.width, self.height = image.size
        image = image.convert('RGB')
        image = np.asarray(image)
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        y, u, v = self.__Rgb2Yuv(r, g, b)
        y_blocks = self.__Encode(y, self.__lt)
        u_blocks = self.__Encode(u, self.__ct)
        v_blocks = self.__Encode(v, self.__ct)
        y_code = self.__Rle(self.__Zig(y_blocks))
        u_code = self.__Rle(self.__Zig(u_blocks))
        v_code = self.__Rle(self.__Zig(v_blocks))
        buff = 0
        tfile = os.path.splitext(filename)[0] + ".gpj"
        if os.path.exists(tfile):
            os.remove(tfile)
        with open(tfile, 'wb') as o:
            o.write(self.height.to_bytes(2, byteorder='big'))
            o.flush()
            o.write(self.width.to_bytes(2, byteorder='big'))
            o.flush()
            o.write((len(y_code)).to_bytes(4, byteorder='big'))
            o.flush()
            o.write((len(u_code)).to_bytes(4, byteorder='big'))
            o.flush()
            o.write((len(v_code)).to_bytes(4, byteorder='big'))
            o.flush()
        self.__Write2File(tfile, y_code, u_code, v_code)

    def __Write2File(self, filename, y_code, u_code, v_code):
        with open(filename, "ab+") as o:
            buff = 0
            bcnt = 0
            data = y_code + u_code + v_code
            for i in range(len(data)):
                if i % 2 == 0:
                    td = data[i]
                    for ti in range(4):
                        buff = (buff << 1) | ((td & 0x08) >> 3)
                        td <<= 1
                        bcnt += 1
                        if bcnt == 8:
                            o.write(buff.to_bytes(1, byteorder='big'))
                            o.flush()
                            buff = 0
                            bcnt = 0
                else:
                    td = data[i]
                    vtl, vts = self.__VLI(td)
                    for ti in range(4):
                        buff = (buff << 1) | ((vtl & 0x08) >> 3)
                        vtl <<= 1
                        bcnt += 1
                        if bcnt == 8:
                            o.write(buff.to_bytes(1, byteorder='big'))
                            o.flush()
                            buff = 0
                            bcnt = 0
                    for ts in vts:
                        buff <<= 1
                        if ts == '1':
                            buff |= 1
                        bcnt += 1
                        if bcnt == 8:
                            o.write(buff.to_bytes(1, byteorder='big'))
                            o.flush()
                            buff = 0
                            bcnt = 0
            if bcnt != 0:
                buff <<= (8 - bcnt)
                o.write(buff.to_bytes(1, byteorder='big'))
                o.flush()
                buff = 0
                bcnt = 0

    def __IDct(self, block):
        # IDCT变换
        res = np.dot(np.transpose(self.__dctA), block)
        res = np.dot(res, self.__dctA)
        return res

    def __IQuantize(self, block, tag):
        res = block
        if tag == self.__lt:
            res *= self.__lq
        elif tag == self.__ct:
            res *= self.__cq
        return res

    def __IFill(self, matrix):
        matrix = matrix[:self.height, :self.width]
        return matrix

    def __Decode(self, blocks, tag):
        tlist = []
        for b in blocks:
            b = np.array(b)
            tlist.append(self.__IDct(self.__IQuantize(b, tag).reshape(8 ,8)))
        height_fill, width_fill = self.height, self.width
        if height_fill % 16 != 0:
            height_fill += 16 - height_fill % 16
        if width_fill % 16 != 0:
            width_fill += 16 - width_fill % 16
        rlist = []
        for hi in range(height_fill // 8):
            start = hi * width_fill // 8
            rlist.append(np.hstack(tuple(tlist[start: start + (width_fill // 8)])))
        matrix = np.vstack(tuple(rlist))
        res = self.__IFill(matrix)
        return res

    def __ReadFile(self, filename):
        with open(filename, "rb") as o:
            tb = o.read(2)
            self.height = int.from_bytes(tb, byteorder='big')
            tb = o.read(2)
            self.width = int.from_bytes(tb, byteorder='big')
            tb = o.read(4)
            ylen = int.from_bytes(tb, byteorder='big')
            tb = o.read(4)
            ulen = int.from_bytes(tb, byteorder='big')
            tb = o.read(4)
            vlen = int.from_bytes(tb, byteorder='big')
            buff = 0
            bcnt = 0
            rlist = []
            itag = 0
            icnt = 0
            vtl, tb, tvtl = None, None, None
            while len(rlist) < ylen + ulen + vlen:
                if bcnt == 0:
                    tb = o.read(1)
                    if not tb:
                        break
                    tb = int.from_bytes(tb, byteorder='big')
                    bcnt = 8
                if itag == 0:
                    buff = (buff << 1) | ((tb & 0x80) >> 7)
                    tb <<= 1
                    bcnt -= 1
                    icnt += 1
                    if icnt == 4:
                        rlist.append(buff & 0x0F)
                    elif icnt == 8:
                        vtl = buff & 0x0F
                        tvtl = vtl
                        itag = 1
                        buff = 0
                else:
                    buff = (buff << 1) | ((tb & 0x80) >> 7)
                    tb <<= 1
                    bcnt -= 1
                    tvtl -= 1
                    if tvtl == 0 or tvtl == -1:
                        rlist.append(self.__IVLI(vtl, bin(buff)[2:].rjust(vtl, '0')))
                        itag = 0
                        icnt = 0
        y_dcode = rlist[:ylen]
        u_dcode = rlist[ylen:ylen+ulen]
        v_dcode = rlist[ylen+ulen:ylen+ulen+vlen]
        return y_dcode, u_dcode, v_dcode

    def __Zag(self, dcode):
        dcode = np.array(dcode).reshape((len(dcode) // 64, 64))
        tz = np.zeros(dcode.shape)
        for i in range(len(self.__zag)):
            tz[:, i] = dcode[:, self.__zag[i]]
        rlist = tz.tolist()
        return rlist

    def __IRle(self, dcode):
        rlist = []
        for i in range(len(dcode)):
            if i % 2 == 0:
                rlist += [0] * dcode[i]
            else:
                rlist.append(dcode[i])
        return rlist

    def Decompress(self, filename):
        y_dcode, u_dcode, v_dcode = self.__ReadFile(filename)
        y_blocks = self.__Zag(self.__IRle(y_dcode))
        u_blocks = self.__Zag(self.__IRle(u_dcode))
        v_blocks = self.__Zag(self.__IRle(v_dcode))
        y = self.__Decode(y_blocks, self.__lt)
        u = self.__Decode(u_blocks, self.__ct)
        v = self.__Decode(v_blocks, self.__ct)
        r = (y + 1.402 * (v - 128))
        g = (y - 0.34414 * (u - 128) - 0.71414 * (v - 128))
        b = (y + 1.772 * (u - 128))
        r = Image.fromarray(r).convert('L')
        g = Image.fromarray(g).convert('L')
        b = Image.fromarray(b).convert('L')
        image = Image.merge("RGB", (r, g, b))
        image.save("./result.bmp", "bmp")
        image.show()

    def __VLI(self, n):
        # 获取整数n的可变字长整数编码
        ts, tl = 0, 0
        if n > 0:
            ts = bin(n)[2:]
            tl = len(ts)
        elif n < 0:
            tn = (-n) ^ 0xFFFF
            tl = len(bin(-n)[2:])
            ts = bin(tn)[-tl:]
        else:
            tl = 0
            ts = '0'

        return tl, ts

    def __IVLI(self, tl, ts):
        # 获取可变字长整数编码的整数值
        if tl != 0:
            n = int(ts, 2)
            if ts[0] == '0':
                n = n ^ 0xFFFF
                n = int(bin(n)[-tl:], 2)
                n = -n
        else:
            n = 0

        return n

dog_img = io.imread("./dog.jpg")
target = dog_img

RLE_code, shape = RLE_encode(target)
compress_rate = RLE_compress_rate(RLE_code, shape)
print(f"RLE Compression Rate: {compress_rate:.2%}")

Huffman_code, code_map, code_length_map, shape = huffman_encode(target)
compress_rate = Huffman_compress_rate(Huffman_code, code_map, shape)
print(f"Huffman Compression Rate: {compress_rate:.2%}")


nc = 10
imgResult = dct_2d_multichannel(target, nc)
idct_img = idct_2d_multichannel(imgResult)
idct_img = np.clip(idct_img, 0, 255).astype(np.uint8)
compress_rate = dct_compress_rate(target, nc)
print(f"DCT Compression Rate: {compress_rate:.2%}")


kjpeg = KJPEG()
original_img = Image.open("./sea.jpg")
kjpeg_img = kjpeg.Compress("./sea.jpg")
dekjpeg_img = kjpeg.Decompress("./sea.gpj")

plt.figure(figsize=(12, 9))
plt.subplot(1, 3, 1)
plt.imshow(target)
plt.title("Original Image")
plt.subplot(1, 3, 2)
plt.imshow(np.clip(imgResult, 0, 255).astype(np.uint8))
plt.title("DCT Image (Visualized)")
plt.subplot(1, 3, 3)
plt.imshow(idct_img)
plt.title("IDCT Image")
plt.show()