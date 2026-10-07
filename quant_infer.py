import onnx
from onnx import numpy_helper
import numpy as np

m = onnx.load("saw2sin_int8.onnx")
P = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}



S_in   = P["s_scale"]                    # shape ()  标量
zp     = P["s_zero_point"]
q_w0   = P["net.0.weight_quantized"]     # shape (64,1) int8
S_w0   = P["net.0.weight_scale"]         # shape (64,)  每通道
b0_int = P["net.0.bias_quantized"]       # shape (64,)  int32
S_out0 = P["/net/net.1/Relu_output_0_scale"]      # 隐藏层输出 scale
z_out0 = P["/net/net.1/Relu_output_0_zero_point"] # -128

S_in2   = P["/net/net.1/Relu_output_0_scale"]                    # shape ()  标量
z_in2   = P["/net/net.1/Relu_output_0_zero_point"]     # shape (64,1) int8
W_q2   = P["net.2.weight_quantized"]         # shape (64,)  每通道
S_w2 = P["net.2.weight_scale"]       # shape (64,)  int32
b2_int = P["net.2.bias_quantized"]
S_out2 = P["/net/net.3/Relu_output_0_scale"]      # 隐藏层输出 scale
z_out2 = P["/net/net.3/Relu_output_0_zero_point"] # -128

S_in3   = P["/net/net.3/Relu_output_0_scale"]                    # shape ()  标量
z_in3   = P["/net/net.3/Relu_output_0_zero_point"]     # shape (64,1) int8
W_q3   = P["net.4.weight_quantized"]         # shape (64,)  每通道
S_w3 = P["net.4.weight_scale"]       # shape (64,)  int32
b3_int = P["net.4.bias_quantized"]
S_out3 = P["y_scale"]      # 隐藏层输出 scale
z_out3 = P["y_zero_point"] # -128

print(P.keys())                       # 看看有哪些名字
print(S_in, type(S_in))               # 确认标量读法
print(q_w0.shape, q_w0.dtype)         # 确认 (64,1) int8

M2 = S_in2*S_w2[0]/S_out2
print(M2)
#第一层

#计算qin输入
# sin  = 0.5
# q_in = np.array([round(float(sin / S_in))],dtype=np.int32)
# print(q_in)
# #acc权重吧？
# acc = q_w0.astype(np.int32) @ q_in + b0_int
# print(acc[0])

# M = S_in * S_w0 / S_out0 #这是在干什么？相当于是自建变量？
# print(M.shape, M[0])

# q_out = np.clip(np.round(acc * M).astype(np.int32) + int(z_out0), -128,127)
# #激活参数
# print(q_out[0])

# h1 = S_out0 * (q_out - int(z_out0))
# print(h1[0])

def quant_layer(q_in, W_q, S_w, b_int, S_in, z_in, S_out, z_out):
    acc = W_q.astype(np.int32) @ (q_in - int(z_in))+ b_int
    M = S_in * S_w / S_out #这是在干什么？相当于是自建变量？
    q_out = np.clip(np.round(acc * M).astype(np.int32) + int(z_out), -128,127)
    h = S_out * (q_out - int(z_out))

    return q_out, h


s  = 0.5
q0 = np.array([round(float(s / S_in))],dtype=np.int32)
q1, h1=quant_layer(q0, q_w0, S_w0, b0_int, float(S_in), int(zp), float(S_out0), float(z_out0))
q2, h2=quant_layer(q1, W_q2, S_w2, b2_int, float(S_in2), int(z_in2), float(S_out2), float(z_out2))
q3, h3 = quant_layer(q2, W_q3, S_w3, b3_int, S_in=float(S_in3), z_in=int(z_in3), S_out=float(S_out3), z_out=int(z_out3))


print(q0)
print(q1, h1)
print(q2, h2)
print(q3, h3)

#第二层



import onnx, onnxruntime as ort

m2 = onnx.load("saw2sin_int8.onnx")
vi = onnx.helper.ValueInfoProto()
vi.name = "/net/net.1/Relu_output_0_QuantizeLinear_Output"   # 第1层输出的 int8 张量
m2.graph.output.append(vi)

sess = ort.InferenceSession(m2.SerializeToString(), providers=["CPUExecutionProvider"])
# h1_q = sess.run([vi.name], {"s": np.array([[0.5]], dtype=np.float32)})[0]
# print(h1_q.shape)          # 大概是 (1, 64)
# print(h1_q[0, :8])         # ORT 的真值
# print(q3[:8])           # 你的实现
# print("max diff =", np.abs(h1_q[0].astype(np.int64) - q_out).max())

y = float(S_out3) * (q3 - int(z_out3))
y_ort = sess.run(["y"], {"s": np.array([[0.5]], dtype=np.float32)})[0]
print(y_ort, y)   # y = S_out*(q3 - z_out) 的最后一个值