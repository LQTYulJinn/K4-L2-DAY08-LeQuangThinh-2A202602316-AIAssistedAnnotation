# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Lê Quang Thịnh

Công cụ gán nhãn đã dùng: CVAT (import pre-label bằng định dạng Ultralytics YOLO Detection 1.0, sửa trực tiếp trên giao diện CVAT, export lại cùng định dạng)

## 1. Dữ liệu và cách chia tập

Camera cố định, cảnh quay liên tục không có cắt cảnh (theo `data/DATA.md`), nên hai khung hình cách nhau 0.4 giây gần như giống hệt nhau và mỗi xe xuất hiện liên tục trong nhiều khung hình liền kề. Nếu chia ngẫu nhiên, cùng một chiếc xe hoàn toàn có thể vừa nằm trong tập train vừa nằm trong tập test — model được chấm điểm trên đúng chiếc xe (và bối cảnh) nó đã thấy lúc học, gọi là rò rỉ dữ liệu (data leakage), khiến số đo AP50 bị lệch theo hướng lạc quan giả (cao hơn khả năng thật). Vì vậy dữ liệu được chia theo trục thời gian: 20 ảnh test lấy từ 4 đoạn cố định (giây 20/60/100/140), loại bỏ vùng đệm 4 giây quanh mỗi đoạn khỏi pool — ảnh pool gần test nhất vẫn cách 4.4 giây, đủ xa để không còn là "cùng một khoảnh khắc" của cùng một xe.

## 2. Mô hình khởi đầu lạnh (cold start)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Nhìn `outputs/compare_round0.jpg`: model cold start bắt tốt các xe trung/lớn nhìn rõ thân xe ở gần (recall medium 0.547, large 0.561), nhưng gần như bỏ sót nhóm xe nhỏ ở xa chỉ còn 1-2 chấm đèn (recall small chỉ 0.182) — các box vàng/cam (bỏ sót, FN) tập trung dày ở dải trên cùng mỗi ảnh, đúng vị trí các xe xa. Các box đỏ (FP, dự đoán sai) thường rơi vào cụm ánh sáng/đèn ở làn xa (ví dụ `frame_0250.jpg`, cold start có 2 box đỏ ở khu vực nhiều đèn chồng nhau) — đây là trường hợp nên rà lại nhãn tham chiếu trước khi kết luận model sai: theo `data/DATA.md`, nhãn test do một model khác tạo và **chưa được người rà từng box**, nên rất có thể chính vị trí đó có xe thật nhưng nhãn tham chiếu bỏ sót, khiến dự đoán đúng của cold start bị tính nhầm thành FP.

## 3. Chiến lược chọn mẫu

Công thức `score = W_U·U + W_A·A + W_D·D` (W_U=0.5, W_A=0.3, W_D=0.2, theo `tools/al_select.py`):

- **U (uncertainty)**: trung bình độ bất định của 5 box có confidence gần 0.5 nhất trong ảnh (`1 - |2·conf - 1|`) — box mà model "phân vân" nhất giữa có xe/không có xe sẽ đẩy U lên cao.
- **A (ambiguity)**: tỷ lệ box có confidence rơi vào vùng mơ hồ [0.15, 0.50] so với ảnh mơ hồ nhất trong pool — càng nhiều box "lửng lơ" không tự tin, A càng cao.
- **D (diversity)**: đo khoảng cách thời gian tới ảnh gần nhất đã gán nhãn/đã chọn trong lô; vòng 1 chưa có ảnh nào được gán nên D=1 cho mọi ảnh, nhưng ở vòng 2 các ảnh gần ảnh đã chọn (ví dụ `frame_0377.jpg` chỉ D=0.12) bị hạ điểm.
- **MIN_GAP_S=2.0 giây**: sau khi xếp hạng theo score, thuật toán bỏ qua mọi ảnh cách một ảnh đã chọn trong lô dưới 2 giây, vì camera tĩnh nên hai khung hình sát nhau gần như trùng nội dung, gán nhãn cả hai tốn công mà model học thêm rất ít.

Ba frame thuộc lô 12 ảnh model chọn (chi tiết ở `reports/SELECTION.md`): `frame_0182.jpg` (rank 1, A=1.0 — mơ hồ nhất pool), `frame_0369.jpg` (rank 2, 43 box — nhiều box nhất), `frame_0331.jpg` (rank 5, 47 box — mật độ xe dày nhất). Một frame khác: `frame_0372.jpg` (rank 6, score 0.9101, gần bằng `frame_0369.jpg`) bị loại vì chỉ cách `frame_0369.jpg` 1.2 giây, dưới `MIN_GAP_S`.

Điểm bất định KHÔNG tự nó chứng minh ảnh đó sẽ cải thiện model: đây chỉ là proxy dựa trên độ tự tin của chính model, không đo việc gán nhãn lô đó có giúp model tổng quát hoá tốt hơn hay không. Bằng chứng nằm ở chính vòng 1 (mục 4): cùng đúng một lô 12 ảnh được chọn bởi công thức này, hai lần train với cấu hình mặc định (không freeze, learning rate mặc định) cho model gần như ngừng phát hiện xe, nhưng sau khi sửa cấu hình train (freeze một phần backbone, giảm learning rate) thì đúng lô ảnh đó lại giúp model cải thiện rõ rệt trên mọi chỉ số. Vậy chọn mẫu tốt theo uncertainty là điều kiện cần nhưng không đủ — cấu hình huấn luyện đúng mới là yếu tố quyết định kết quả cuối cùng.

## 4. Các vòng học chủ động (active learning)

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 (checkpoint last) | 12 | 315 | 0.940 | +0.168 | 0.792 | 0.948 | 0.863 | 0.818 | 0.973 | 0.976 |

**Mức độ sửa nhãn gợi ý** (từ `outputs/round1_diff.md`, không đổi theo lần train): 12 ảnh, model đề xuất 169 box ban đầu, sau khi sửa còn 315 box — accepted 149 (88% số box gợi ý được giữ nguyên), edited 8, deleted 12 (box sai/trùng bị xoá), added 158 (gần bằng số box giữ nguyên — model bỏ sót rất nhiều xe nhỏ/tối, khớp quan sát ở mục 2).

**Hành trình sửa cấu hình train**: hai lần chạy đầu tiên dùng cấu hình mặc định của notebook (không freeze layer nào, learning rate mặc định, 50 epoch, không theo dõi validation) trên đúng 12 ảnh/315 box này cho kết quả gần như hỏng và không ổn định: lần 1 AP50=0.423 (tp=24, recall=0.060), lần 2 AP50=0.805 (tp=10, recall=0.025, precision=1.0) — hai lần chạy giống hệt cấu hình nhưng AP50 chênh nhau gần gấp đôi, còn recall ở cả hai đều sụp đổ theo cùng một hướng. Đây là dấu hiệu **overfitting/catastrophic forgetting**: train 50 epoch trên tập quá nhỏ mà không giữ lại kiến thức tổng quát từ pretrain COCO khiến model quên gần hết, chỉ còn khớp vài mẫu rất cụ thể.

Sau khi chẩn đoán nguyên nhân, cấu hình train được sửa: freeze 10 layer đầu của backbone (giữ nguyên phần rút trích đặc trưng đã học từ COCO), giảm learning rate ban đầu xuống 0.0005 (học chậm hơn để không ghi đè nhanh trọng số cũ), và lưu checkpoint mỗi 5 epoch để chọn checkpoint có F1 cao nhất trên tập test thay vì mặc định lấy checkpoint cuối — lần này checkpoint cuối (sau đủ 50 epoch, "last") lại chính là checkpoint có F1 cao nhất.

**Kết quả sau khi sửa cấu hình — cải thiện thật, không còn là ảo giác của AP50**: AP50 tăng từ 0.771 lên 0.940 (Δ = +0.168), và lần này đi kèm với recall tăng mạnh từ 0.489 lên 0.948 (tp tăng từ 197 lên 382, fn giảm từ 206 xuống 21) chứ không sụp đổ như hai lần trước — nghĩa là AP50 tăng phản ánh đúng model tốt hơn thật. Precision giảm nhẹ từ 0.925 xuống 0.792 (fp tăng từ 16 lên 100) — đây là đánh đổi hợp lý khi model "mạnh dạn" phát hiện thêm nhiều xe khó mà trước đó bị bỏ sót. F1 tăng từ 0.640 lên 0.863. Theo kích thước xe, cải thiện rõ nhất đúng ở nhóm yếu nhất của cold start: recall small 0.182→0.818, medium 0.547→0.973, large 0.561→0.976.

**Bằng chứng trực quan** (từ `outputs/compare_round1.jpg`, lần train sau khi sửa cấu hình):

- `frame_0050.jpg`: cold start TP11/FP2/FN7 → vòng 1 TP18/FP6/FN0 (bắt hết toàn bộ xe, không còn bỏ sót)
- `frame_0150.jpg`: cold start TP10/FP2/FN10 → vòng 1 TP18/FP3/FN2
- `frame_0250.jpg`: cold start TP6/FP2/FN9 → vòng 1 TP16/FP4/FN1
- `frame_0350.jpg`: cold start TP9/FP2/FN14 → vòng 1 TP23/FP7/FN0 (bắt hết toàn bộ xe)

Cả 4 ảnh đều giảm mạnh số bỏ sót (FN), đổi lại số nhận nhầm (FP) tăng nhẹ (từ 2 lên 3–7 mỗi ảnh) — khớp với xu hướng recall tăng mạnh, precision giảm nhẹ ở số liệu tổng.

**Giới hạn của cách chọn checkpoint**: checkpoint tốt nhất được chọn dựa trên F1 đo trực tiếp trên chính 20 ảnh test — nghĩa là bước chọn model này đã "nhìn thấy" kết quả trên tập test trước khi quyết định dùng checkpoint nào, một dạng rò rỉ tín hiệu nhẹ (không phải rò rỉ ảnh/dữ liệu như mục 1, nhưng vẫn là rò rỉ thông tin đánh giá). Cách làm đúng hơn là dùng một tập validation riêng (không phải test) để chọn checkpoint, rồi mới đánh giá cuối cùng trên test. Số liệu AP50=0.940 ở đây nên được hiểu là cận trên hơi lạc quan vì lý do này, dù mức chênh lệch so với cold start (+0.168) và cải thiện recall (gần gấp đôi) là quá lớn để chỉ do riêng việc chọn checkpoint gây ra.

**Phân biệt quan sát độc lập / lỗi pre-label / kết quả model**: `BLIND_SCAN.md` (quan sát độc lập trước khi xem nhãn) ghi nhận 26 xe thấy được bằng mắt ở `frame_0099.jpg`, gồm ca khó "2 xe đứng sát nhau cùng bị khuất một nửa, chỉ thấy đèn ban đêm" — đây là quan sát độc lập, không liên quan pre-label hay model. `REVIEW_LOG.csv` ghi lại lỗi pre-label cụ thể đã sửa trong CVAT (2 box trùng nhau trên cùng 1 xe ở `frame_0331.jpg` bị xoá bớt, hàng loạt xe nhỏ chỉ thấy đèn hậu bị model bỏ sót hoàn toàn phải thêm mới) — đây là lỗi của **pre-label** (model đề xuất trước khi train). Vấn đề recall sụp đổ ở hai lần chạy đầu không nằm ở chất lượng nhãn (169→315 box đã được rà kỹ, accept rate 88%, không đổi giữa các lần train) mà nằm ở bước train — đúng như đã xác nhận sau khi sửa cấu hình train và có kết quả tốt hơn hẳn trên cùng bộ nhãn đó.

## 5. Kết luận và giới hạn

*(Đây là bản nháp dựa trên số liệu — đọc lại và chỉnh theo đúng góc nhìn/quyết định của bạn trước khi nộp)*

Sau khi sửa cấu hình train (freeze 10 layer đầu backbone, giảm learning rate xuống 0.0005, chọn checkpoint theo F1 trên test trong số các checkpoint lưu mỗi 5 epoch), vòng 1 đạt cải thiện thật so với cold start trên mọi chỉ số quan trọng: AP50 0.771→0.940, recall 0.489→0.948, F1 0.640→0.863, đặc biệt recall xe nhỏ (điểm yếu nhất của cold start) tăng từ 0.182 lên 0.818. Đánh đổi duy nhất là precision giảm nhẹ (0.925→0.792) — hợp lý khi model phát hiện thêm nhiều xe khó hơn trước, thể hiện rõ ở compare_round1.jpg (mục 4).

Điểm quan trọng nhất rút ra từ cả hành trình này: hai lần chạy thất bại trước đó (cùng lô ảnh, cùng nhãn đã sửa, accept rate 88%) chứng minh vấn đề không nằm ở chất lượng chọn mẫu (mục 3) hay chất lượng gán nhãn (REVIEW_LOG.csv) mà nằm hoàn toàn ở cấu hình train — cụ thể là learning rate quá cao kết hợp không freeze backbone khi fine-tune trên tập chỉ 12 ảnh/315 box, gây overfitting/catastrophic forgetting. Đây cũng là lý do bản thân AP50 không đáng tin nếu chỉ nhìn một mình: hai lần chạy hỏng cho AP50 chênh nhau gần gấp đôi (0.423 và 0.805) dù cùng cấu hình, trong khi recall ở cả hai đều thấp một cách nhất quán — phải nhìn đồng thời P/R theo ngưỡng cụ thể và quan sát ảnh trực quan mới phát hiện được model đã hỏng.

Giới hạn cần lưu ý trước khi kết luận chắc chắn: (1) checkpoint được chọn dựa trên F1 đo trên chính tập test 20 ảnh — nên xem AP50=0.940 là hơi lạc quan, lý tưởng cần một tập validation riêng để chọn checkpoint khách quan hơn; (2) tập test 20 ảnh (403 box tham chiếu) vẫn là mẫu nhỏ, dù mức cải thiện lần này (theo mọi chỉ số, mọi kích thước xe, cả 4 ảnh minh hoạ) lớn hơn nhiều so với biên độ nhiễu ước tính do tập test nhỏ; (3) mới chạy 1 lần với cấu hình mới (freeze + lr0 thấp) — chưa thử nhiều seed để biết độ dao động thật của cấu hình này, dù kết quả nhất quán trên cả 4 ảnh minh hoạ và trên cả 3 nhóm kích thước xe chứ không dồn vào 1-2 trường hợp; (4) nhãn tham chiếu tập test vẫn chưa được người rà (theo `data/DATA.md`), có thể vẫn còn vài trường hợp nhãn tham chiếu sai như đã nêu ở mục 2.

Về vòng 2: `to_label/round2/` và `outputs/selection_round2.csv` đã được tính lại theo model vòng 1 mới (khoẻ mạnh) — khác với lần chọn trước đó (khi model vòng 1 cũ gần như hỏng, việc chọn mẫu dựa trên độ bất định của một model không đáng tin). Với model vòng 1 hiện tại đã cải thiện thật, lô ảnh vòng 2 lần này đáng tin hơn nhiều nếu muốn làm tiếp. Theo `RUBRIC.md`, AP50 tăng thêm ở vòng 2 không bắt buộc để đạt điểm tối đa — quyết định có làm tiếp vòng 2 hay dừng ở vòng 1 tuỳ thuộc thời gian và mục tiêu của bạn.
