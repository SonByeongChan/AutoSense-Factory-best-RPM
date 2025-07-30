const hours = [...Array(24).keys()].map(h => h + '시');
const weekLabels = ['월', '화', '수', '목', '금', '토', '일'];
const monthLabels = ['1주차', '2주차', '3주차', '4주차', '5주차', '6주차'];

let selectedDate = null;
let chartRefs = {};

// ▶ 공통 차트 그리기 함수
function drawBar(id, label, data, color) {
  const ctx = document.getElementById(id).getContext('2d');
  if (chartRefs[id]) chartRefs[id].destroy();

  const labels = id.startsWith("week") ? weekLabels :
                 id.startsWith("month") ? monthLabels : hours;

  chartRefs[id] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label, data, backgroundColor: color }]
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: '#ccc' }, grid: { color: '#444' } },
        y: { beginAtZero: true, ticks: { color: '#ccc' }, grid: { color: '#444' } }
      },
      plugins: { legend: { labels: { color: '#fff' } } }
    }
  });
}

// ▶ 일간 데이터 불러오기 및 렌더링
async function fetchDayData() {
  const res = await fetch(`/api/production_day?date=${selectedDate}`);
  const json = await res.json();
  drawBar('chartProduction', '생산량 (g)', json.production_by_hour, 'skyblue');
  drawBar('chartLoss', '손실률 (g)', json.loss_by_hour, 'red');
  drawBar('chartRevenue', '매출액 (₩)', json.revenue_by_hour, 'limegreen');
  renderDailyTable(json);
}

// ▶ 주간 데이터
async function fetchWeekData() {
  const res = await fetch(`/api/production_week?date=${selectedDate}`);
  const data = await res.json();
  drawBar('weekProductionChart', '생산량 (g)', data.production_by_day, 'skyblue');
  drawBar('weekLossChart', '손실률 (g)', data.loss_by_day, 'red');
  drawBar('weekRevenueChart', '매출액 (₩)', data.revenue_by_day, 'limegreen');
  renderWeeklyTable(data);
}

// ▶ 월간 데이터
async function fetchMonthData() {
  const res = await fetch(`/api/production_month?date=${selectedDate}`);
  const data = await res.json();
  drawBar('monthProductionChart', '생산량 (g)', data.production_by_week, 'skyblue');
  drawBar('monthLossChart', '손실률 (g)', data.loss_by_week, 'red');
  drawBar('monthRevenueChart', '매출액 (₩)', data.revenue_by_week, 'limegreen');
  renderMonthlyTable(data);
}

// ✅ CSV 전체 저장 함수
function exportToCSV(filename, dataArray) {
  const csvRows = [];
  const headers = Object.keys(dataArray[0]);
  csvRows.push(headers.join(','));

  for (const row of dataArray) {
    const values = headers.map(header => JSON.stringify(row[header] ?? ''));
    csvRows.push(values.join(','));
  }

  const BOM = '\uFEFF'; // ✅ Excel 호환을 위한 BOM
  const blob = new Blob([BOM + csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
}
   

// ▶ 일간 CSV 전체 저장
function exportDayCSV() {
  const date = document.getElementById('selected-date').value;
  fetch(`/api/production_day?date=${date}`)
    .then(res => res.json())
    .then(data => {
      const fullData = [];
      for (let i = 0; i < 24; i++) {
        fullData.push({
          시간: i + "시",
          생산량: data.production_by_hour[i],
          불량률: data.defect_by_hour[i],
          손실률: data.loss_by_hour[i],
          매출액: data.revenue_by_hour[i]
        });
      }
      exportToCSV(`일간현황_${date}.csv`, fullData);;
    });
}

function exportWeekCSV() {
  const date = document.getElementById('selected-date').value;
  fetch(`/api/production_week?date=${date}`)
    .then(res => res.json())
    .then(data => {
      const fullData = [];

      const selected = new Date(date);
      const weekStart = new Date(selected);
      weekStart.setDate(selected.getDate() - selected.getDay() + 1);

      for (let i = 0; i < 7; i++) {
        const day = new Date(weekStart);
        day.setDate(weekStart.getDate() + i);
        const label = formatWeekday(day);

        fullData.push({
          날짜: label,
          생산량: data.production_by_day[i],
          불량률: data.defect_by_day[i],
          손실률: data.loss_by_day[i],
          매출액: data.revenue_by_day[i]
        });
      }

      exportToCSV(`주간현황_${date}.csv`, fullData);
    });
}


// ▶ 월간 CSV 전체 저장
function exportMonthCSV() {
  const date = document.getElementById('selected-date').value;
  fetch(`/api/production_month?date=${date}`)
    .then(res => res.json())
    .then(data => {
      const fullData = [];
      for (let i = 0; i < 6; i++) {
        fullData.push({
          주차: `${i + 1}주차`,
          생산량: data.production_by_week[i],
          불량률: data.defect_by_week[i],
          손실률: data.loss_by_week[i],
          매출액: data.revenue_by_week[i]
        });
      }
      exportToCSV(`월간현황_${date}.csv`, fullData);
    });
}

// ✅ 테이블 렌더링: 0값 제외

function renderDailyTable(data) {
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = '';
  for (let i = 0; i < 24; i++) {
    const prod = data.production_by_hour[i];
    const loss = data.loss_by_hour[i];
    const defect = data.defect_by_hour[i];
    const revenue = data.revenue_by_hour[i];
    if (prod === 0 && loss === 0 && defect === 0 && revenue === 0) continue;
    const row = document.createElement('tr');
    row.innerHTML = `<td>${i}시</td><td>${prod}</td><td>${defect}</td><td>${loss}</td><td>${revenue}</td>`;
    tbody.appendChild(row);
  }
}

function formatWeekday(dateObj) {
  const weekdayNames = ['일', '월', '화', '수', '목', '금', '토'];
  const yyyy = dateObj.getFullYear();
  const mm = String(dateObj.getMonth() + 1).padStart(2, '0');
  const dd = String(dateObj.getDate()).padStart(2, '0');
  const weekday = weekdayNames[dateObj.getDay()];
  return `${yyyy}-${mm}-${dd} (${weekday})`;
}

function renderWeeklyTable(data) {
  const tbody = document.querySelector('#tab-content-week tbody');
  tbody.innerHTML = '';

  // 📌 선택된 날짜로부터 해당 주의 월요일 계산
  const selected = new Date(selectedDate);
  const weekStart = new Date(selected);
  weekStart.setDate(weekStart.getDate() - weekStart.getDay() + 1);  // 월요일 기준

  for (let i = 0; i < 7; i++) {
    const currentDate = new Date(weekStart);
    currentDate.setDate(weekStart.getDate() + i);  // i일 더한 날짜

    const label = formatWeekday(currentDate);  // "YYYY-MM-DD (요일)" 형식

    const prod = data.production_by_day[i];
    const loss = data.loss_by_day[i];
    const defect = data.defect_by_day[i];
    const revenue = data.revenue_by_day[i];

    if (prod === 0 && loss === 0 && defect === 0 && revenue === 0) continue;

    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${label}</td>
      <td>${prod}</td>
      <td>${defect}</td>
      <td>${loss}</td>
      <td>${revenue}</td>
    `;
    tbody.appendChild(row);
  }
}

function renderMonthlyTable(data) {
  const tbody = document.querySelector('#tab-content-month tbody');
  tbody.innerHTML = '';
  for (let i = 0; i < 6; i++) {
    const prod = data.production_by_week[i];
    const loss = data.loss_by_week[i];
    const defect = data.defect_by_week[i];
    const revenue = data.revenue_by_week[i];
    if (prod === 0 && loss === 0 && defect === 0 && revenue === 0) continue;
    const row = document.createElement('tr');
    row.innerHTML = `<td>${monthLabels[i]}</td><td>${prod}</td><td>${defect}</td><td>${loss}</td><td>${revenue}</td>`;
    tbody.appendChild(row);
  }
}
 
// ✅ 탭 및 날짜 처리
function setTab(el, tabId) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.querySelectorAll('.tab-sheet').forEach(c => c.style.display = 'none');
  document.getElementById('tab-content-' + tabId).style.display = 'block';
}

function onDateChange() {
  const input = document.getElementById("selected-date");
  if (!input || !input.value) {
    alert("날짜를 선택해주세요.");
    return;
  }
  selectedDate = input.value;
  fetchDayData();
  fetchWeekData();
  fetchMonthData(); 
}

document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("selected-date");
  if (!input.value) input.value = new Date().toISOString().slice(0, 10);
  selectedDate = input.value;
  if (selectedDate) {
    setTab(document.querySelector('.tab.active'), 'day');
    fetchDayData();
    fetchWeekData();
    fetchMonthData();
  }
});
