import pdfplumber
import openpyxl

# 创建一个Excel工作簿
wb = openpyxl.Workbook()
sheet = wb.active

# 加载PDF文件
with pdfplumber.open("C:/Users/Administrator/Desktop/bc.pdf") as pdf:
    # 假设PDF只有一页
    page = pdf.pages[0]

    print(page)

    # 查找页面上的表格
    for table in page.extract_tables():
        for row in table:
            print(row)
            sheet.append(row)

# 保存Excel文件
wb.save("C:/Users/Administrator/Desktop/bc.xlsx")