# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, năm frame ưu tiên nếu chỉ có ngân sách rà năm ảnh:

1. **frame_0182.jpg** — rank 1, score 0.9591 (U=0.9182, A=1.0, D=1.0), t=72.8s, 28 box. A=1.0 (tối đa toàn pool) nghĩa là tỷ lệ box có confidence rơi vào vùng mơ hồ [0.15, 0.50] cao nhất — ảnh gây tranh cãi nhất cho model.
2. **frame_0369.jpg** — rank 2, score 0.9324 (U=0.9315, A=0.8889, D=1.0), t=147.6s, 43 box — nhiều box nhất trong lô, U rất cao.
3. **frame_0331.jpg** — rank 5, score 0.9154, 47 box — mật độ xe/box dày đặc nhất, tối đa hoá thông tin thu được mỗi lượt gán nhãn.
4. **frame_0392.jpg** — rank 15 (thấp hơn 3 ảnh trên) nhưng U=0.9747, cao nhất toàn lô — model gần như "tung đồng xu" ở ảnh này dù A chỉ 0.6667.
5. **frame_0187.jpg** — rank 10, cân bằng cả 3 chỉ số (U=0.8324, A=0.9444, D=1.0) — đại diện tốt cho các frame còn lại.

Quyết định có xét ảnh gần trùng: giữa `frame_0369.jpg` (rank 2, đã chọn) và `frame_0368.jpg` (rank 9, chỉ cách 0.4s), lô chỉ lấy 1 trong 2 vì `MIN_GAP_S=2.0s` — hai ảnh gần như trùng nội dung do camera tĩnh, gán cả hai tốn công mà model học thêm rất ít.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV: `frame_0182.jpg` (rank 1), `frame_0369.jpg` (rank 2), `frame_0331.jpg` (rank 5) — lý do U/A/D ở trên, lấy từ `outputs/selection_round1.csv`.

Một frame điểm cao nhưng không được chọn: `frame_0372.jpg` (rank 6, score 0.9101, gần bằng `frame_0369.jpg`) — bị loại vì chỉ cách `frame_0369.jpg` (đã chọn) 1.2 giây, dưới ngưỡng `MIN_GAP_S`, dù điểm gần như ngang nhau.

Một frame điểm thấp vẫn nên xem: `frame_0002.jpg` (rank 18, score 0.8658, t=0.8s, gần đầu video) không nằm trong 12 ảnh được chọn, nhưng đáng xem vì là frame sớm nhất có điểm cao — các frame được chọn tập trung ở đoạn 72–157 giây, nên frame đầu video đại diện cho điều kiện ánh sáng/góc quay khác, giúp kiểm tra model có yếu riêng ở đoạn đầu không.

Điểm chọn mẫu không tự nó chứng minh chất lượng mô hình cuối cùng: điểm U/A/D chỉ đo mức độ "phân vân" của model dựa trên confidence tại thời điểm chọn, không đo việc gán nhãn đúng lô đó có giúp model tổng quát hoá tốt hơn hay không. Minh chứng rõ nhất: đúng lô 12 ảnh này (chọn theo U/A cao nhất pool, nhãn đã sửa kỹ, accept rate 88%), hai lần train đầu với cấu hình mặc định (không freeze backbone, learning rate mặc định, 50 epoch) cho model gần như ngừng phát hiện xe (AP50 0.423 và 0.805 nhưng recall chỉ 0.060 và 0.025). Chỉ sau khi sửa cấu hình train (freeze 10 layer đầu backbone, giảm learning rate xuống 0.0005, chọn checkpoint theo F1) thì đúng lô ảnh và nhãn đó mới giúp model cải thiện thật (AP50 0.940, recall 0.948 — chi tiết ở `reports/REPORT.md` mục 4). Vậy chọn mẫu tốt theo uncertainty là điều kiện cần nhưng không đủ: cấu hình huấn luyện đúng mới quyết định lô ảnh được chọn có thực sự phát huy tác dụng hay không.
