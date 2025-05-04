import os
import pandas as pd

def check_and_clean_dataset(file_path, text_column="desc", output_path="cleaned_news_collection.csv"):
    """
    檢查並清理資料集，包括文件是否存在、列名、缺失值和文本長度分佈。
    清理後的資料集將保存到指定的輸出路徑。
    
    :param file_path: 資料集文件的路徑
    :param text_column: 包含文本數據的列名
    :param output_path: 清理後資料集的保存路徑
    """
    # 檢查文件是否存在
    if not os.path.exists(file_path):
        print(f"錯誤: 文件 {file_path} 不存在")
        return
    
    print(f"文件 {file_path} 存在，開始檢查格式...")
    
    # 加載資料集
    try:
        data = pd.read_csv(file_path)
        print("資料集加載成功")
    except Exception as e:
        print(f"錯誤: 無法加載資料集，原因: {e}")
        return
    
    # 檢查列名
    print("資料集的列名：", data.columns.tolist())
    if text_column not in data.columns:
        print(f"錯誤: 指定的文本列 '{text_column}' 不存在於資料集中")
        return
    print(f"文本列 '{text_column}' 存在")
    
    # 檢查缺失值
    missing_count = data[text_column].isnull().sum()
    print(f"文本列 '{text_column}' 中有 {missing_count} 個缺失值")
    if missing_count > 0:
        print("刪除缺失值...")
        data = data.dropna(subset=[text_column])
        print(f"刪除缺失值後，剩餘 {len(data)} 條數據")
    
    # 檢查文本長度分佈
    data["text_length"] = data[text_column].apply(lambda x: len(str(x)))
    print("文本長度的統計信息：")
    print(data["text_length"].describe())
    
    # 保存清理後的資料集
    data = data[[text_column]]  # 只保留文本列
    data.to_csv(output_path, index=False)
    print(f"清理後的資料集已保存到 {output_path}")
    
    return data

# 測試函數
if __name__ == "__main__":
    # 替換為你的資料集路徑
    dataset_path = "C:/Users/user/Desktop/RoBERTa_Model_Selfdesign/DataSet/news_collection.csv"
    text_column_name = "desc"  # 使用 desc 作為文本列
    output_file_path = "C:/Users/user/Desktop/RoBERTa_Model_Selfdesign/DataSet/cleaned_news_collection.csv"
    
    cleaned_data = check_and_clean_dataset(dataset_path, text_column=text_column_name, output_path=output_file_path)
    
    if cleaned_data is not None:
        print("資料集清理完成，清理後的資料集已保存")