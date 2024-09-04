import os

import pandas as pd


class DatabaseHandler:
    def __init__(self):
        self.data_root = './dataset'
        if os.path.isfile('./dataset/original.csv'):
            self.db = pd.read_csv('./dataset/original.csv')
        else:
            self.db = pd.DataFrame(columns=['id', 'paper_name', 'to_vector'])

    def check_db(self):
        # columns = id, paper_name, to_vector(true/false)

        # dataset 폴더 내에서 확장자가 .pdf인 파일만 가져오기
        papers = [paper for paper in os.listdir(self.data_root) if paper.endswith('.pdf')]
        update_papers = []

        # 파일 목록에서 paper_name과 비교할 때 모두 소문자로 통일
        db_paper_names = self.db['paper_name'].str.lower().tolist()
        for paper in papers:
            if paper.lower() in db_paper_names:
                print(f'[{paper}]는 이미지 vector database에 존재하는 자료입니다. 업데이트 목록에서 제외합니다.')
            else:
                update_papers.append(paper)
                print(f'새로운 자료 [{paper}] 준비를 완료하였습니다.')

        return update_papers

    def update_db(self):
        update_papers = self.check_db()

        # 기존 데이터베이스에서 가장 큰 id 값을 찾음
        if not self.db.empty:
            max_id = self.db['id'].max()
        else:
            max_id = 0

        for update_paper in update_papers:
            temp = pd.DataFrame([[max_id+1, update_paper, False]], columns=['id', 'paper_name', 'to_vector'])
            self.db = pd.concat([self.db, temp], ignore_index=True)

            max_id += 1

        self.db.to_csv(f'{self.data_root}/original.csv', index=False)
        print("데이터베이스가 업데이트되었습니다.")



if __name__ == '__main__':
    db_handler = DatabaseHandler()
    db_handler.update_db()
