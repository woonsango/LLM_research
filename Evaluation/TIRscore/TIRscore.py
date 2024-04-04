#Token Intersection Ratio Score

def self_intersection_n(dataset, n, seed=0):
    """
    train data 용
    자기 자신을 n 분할하여 교집합 토큰을 구한다. 개수는 세지 않는다
    dataset: dataset 객체를 받는다
    n: 분할수
    seed: 데이터 random seed로 섞는다
    """
    # 데이터셋을 섞고 데이터 개수를 센다.
    dataset = dataset.shuffle(seed=seed)
    l = dataset.num_rows

    # 데이터개수를 n개로 슬라이싱하여 리스트에 담는다. 리스트에 담는 이유는 for문 돌릴려고
    # 빈 토큰 set을 n개만큼 생성하여 리스트에 담는다.
    data_split = [dataset[l*i//n:l*(i+1)//n] for i in range(n)]
    token_split = [set() for i in range(n)]

    # 슬라이싱된 집합을 돌면서 token들을 set에 넣는다
    for k in range(n):
        embeddings = tokenizer(data_split[k]['text'])['input_ids']
        for i in embeddings:
            for j in i:
                token_split[k].add(j)

    # 모든 set들의 교집합을 구히여 self 교집합을 구한다.
    self_intersection = token_split[0]
    for i in range(1,n):
        self_intersection = self_intersection & token_split[i]

    # 교집합을 반환한다.
    return self_intersection

def core_token_counter_n(dataset, intersection, n, seed=0):
    """
    valid data 용
    자기 자신을 n 분할하여 교집합 토큰을 구하고, 전체 교집합을 제한다. 개수도 저장한다.
    dataset: dataset 객체를 받는다
    intersection: 전체 교집합. self 교집합에서 제외하기 위해 받는다.
    n: 분할수 
    seed: 데이터 random seed로 섞는다
    """
    # 데이터셋을 섞고 데이터 개수를 센다.
    dataset = dataset.shuffle(seed=seed)
    l = dataset.num_rows
    
    # 토큰 개수를 저장할 딕셔너리를 생성한다.
    token_counter = {}
    # 코어 토큰만 저장할 딕셔너리를 생성한다.
    core = {}

    # 데이터개수를 n개로 슬라이싱하여 리스트에 담는다.
    # 빈 토큰 set을 n개만큼 생성하여 리스트에 담는다.
    data_split = [dataset[l*i//n:l*(i+1)//n] for i in range(n)]
    token_split = [set() for i in range(n)]

    # 슬라이싱된 데이터를 돌면서 set에도 넣으면서 빈도수도 딕셔너리에 저장한다.
    for k in range(n):
        embeddings = tokenizer(data_split[k]['text'])['input_ids']
        for i in embeddings:
            for j in i:
                token_split[k].add(j)
                if j in token_counter:
                    token_counter[j] += 1
                else:
                    token_counter[j] = 1

    # 분할 set끼리의 교집합을 구하여 self 교집합을 구한다.
    self_intersection = token_split[0]
    for i in range(1,n):
        self_intersection = self_intersection & token_split[i]

    # 딕셔너리를 돌면서 self 교집함에 존재하면 core 딕셔너리에 추가한다 (원래 딕셔너리에서 삭제하는 방법은 for문을 망가지게 한다)
    for i in token_counter:
        if i in self_intersection:
            core[i] = token_counter[i]

    # 코어 token과 빈도수를 가진 딕셔너리를 반환
    return core

def compute_TRS(train1, train2, valid, n=3, seed=0):
    """
    점수를 계산한다.
    train1: 집단 1 데이터셋
    train2: 집단 2 데이터셋
    valid: 확인하고싶은 데이터셋
    n: 분할 수
    seed: 데이터 shuffle 랜덤 시드
    """
    start = time()
    # 집단 1,2 에 대해 self 교집합을 구한다.
    train1_intersection = self_intersection_n(train1, n, seed)
    train2_intersection = self_intersection_n(train2, n, seed)

    # 집단 1,2 self 교집합끼리의 교집합을 구해 전체 교집합을 구한다.
    intersection = train1_intersection & train2_intersection

    # 집단 1,2의 self 교집합에서 전체 교집합을 구해 각각의 core token을 구한다
    train1_core = train1_intersection - intersection
    train2_core = train2_intersection - intersection

    # 확인할 데이터셋의 core token을 구한다
    val_core = core_token_counter_n(valid, intersection, n, seed)

    # 점수 초기화
    score1 = 0
    score2 = 0

    # 확인할 데이터셋의 core token에서 token을 하나씩 꺼내와 집단 1,2 중 어느쪽에 존재하는지 확인한다. 없다면 패스 있다면 점수를 1점 올려준다.
    for i in val_core:
        if i in train1_core:
            score1 += 1
        elif i in train2_core:
            score2 += 1
    print(score1, score2)
    score1 = score1*100/(score1+score2)
    score2 = score2*100/(score1+score2)
    
    # 출력, 모종의 오류를 확인하기 위해 정답을 2번 그룹으로 두었다.
    print(f"n: {n} seed: {seed} | group1: {score1}% | group2: {score2}")
    return score1, score2