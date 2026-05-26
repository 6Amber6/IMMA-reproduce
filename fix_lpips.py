"""
修复 eval.py 的 LPIPS bug:
1. 修改 img_to_img_similarity 让 LPIPS 支持不同数量的 src/gen 图
2. 删除 main() 里的 assert
"""

# 读原文件
with open('eval/eval.py', 'r') as f:
    code = f.read()

# 修复 1: 替换 img_to_img_similarity 函数中的 lpips 部分
old_lpips_code = """    def img_to_img_similarity(self, src_images, generated_images):
        if self.metric == "lpips":
            return 1.0 - torch.mean(self.model(src_images, generated_images))
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()"""

new_lpips_code = """    def img_to_img_similarity(self, src_images, generated_images):
        if self.metric == "lpips":
            # 修复:允许不同数量的 src 和 gen,计算所有 pairs 的平均 LPIPS
            n_src = src_images.shape[0]
            n_gen = generated_images.shape[0]
            total_lpips = 0.0
            count = 0
            for i in range(n_src):
                for j in range(n_gen):
                    src_i = src_images[i:i+1]
                    gen_j = generated_images[j:j+1]
                    total_lpips += self.model(src_i, gen_j).item()
                    count += 1
            return torch.tensor(1.0 - total_lpips / count)
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()"""

if old_lpips_code in code:
    code = code.replace(old_lpips_code, new_lpips_code)
    print("✓ Fixed img_to_img_similarity for LPIPS")
else:
    print("✗ Could not find img_to_img_similarity (already fixed?)")

# 修复 2: 删除 main() 里的 assert
old_assert = """        if metric == "lpips":
            n_ref = len(list(files_reference))
            n_imma = len(list(files_imma))
            assert n_ref == n_imma"""

new_assert = """        # LPIPS 数量不必相等(已通过 img_to_img_similarity 修复)
        # if metric == "lpips":
        #     n_ref = len(list(files_reference))
        #     n_imma = len(list(files_imma))
        #     assert n_ref == n_imma"""

if old_assert in code:
    code = code.replace(old_assert, new_assert)
    print("✓ Removed LPIPS assert")
else:
    print("✗ Could not find LPIPS assert (already removed?)")

# 写回
with open('eval/eval.py', 'w') as f:
    f.write(code)

print("\nDone! Verify with: diff eval/eval.py.backup eval/eval.py")
