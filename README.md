# sg_transfer_s
比赛信息：http://task.www.sogou.com/cips-sogou_qa/

预训练词向量来自 https://github.com/Embedding/Chinese-Word-Vectors 中提供的知乎问答（Word + Character + Ngram）词向量

结果评测在view_board.ipynb中展示

### 数据格式
'query_id': 10000
'answer': '林海音'
'type': 'factoid'
'query': '《窃读记》的作者的作品'
'passages': [
{'passage_id': 1, 
'passage_text': '原句句式杂糅。改为：《窃读记》是台湾作家林海音的作品。或：《窃读记》的作者是台湾作家林海音。', 
'url': 'https://zhidao.baidu.com/question/1689685153788223788'}, 
{'passage_id': 2, 
'passage_text': '《窃读记》的作者林海音的原名林海音简介\u3000林海音，女，原名林含英，原籍台湾省苗栗县，林海音于１９１８年３月１８日生于日本大阪，不久即返台，当时台湾已被日本帝国主义侵占，其父林焕父不甘在日寇铁蹄下生活，举家迁居北京，小英子即在北京长大。．．．', 'url': 'http://wenwen.sogou.com/z/q1700885258.htm'}, 
{'passage_id': 3, 'passage_text': '《窃读记》的作者林海音的原名林海音简介\u3000林海音，女，原名林含英，原籍台湾省苗栗县，林海音于１９１８年３月１８日生于日本大阪，不久即返台，当时台湾已被日本帝国主义侵占，其父林焕父不甘在日寇铁蹄下生活，举家迁居北京，小英子即在北京长大。．．．', 'url': 'http://zhidao.baidu.com/question/81040161'}, 
{'passage_id': 4, 'passage_text': '课文以“窃读”为线索，以放学后“我”急匆匆地赶到书店，到晚上依依不舍离开的时间顺序和藏身于众多顾客，借雨天读书两个场景的插入，细腻生动地描绘了“窃读”的独特感受与复杂滋味，表现了“我”对读书的热爱和对知识的渴望。\u3000．．．\u3000作者善于通过自语式的独白描绘心境，表达自己的感情。在“窃读”这种氛围中，一方面享受阅读的快乐，一方面还要时刻关注周围的环境，非常形象生动地表现了“我”的心情变化，使人如历其境。．．．', 'url': 'http://wenwen.sogou.com/z/q714108600.htm'}
]
}
