const { chromium } = require('@playwright/test');

async function run() {
  console.log("Launching browser...");
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 }
  });
  const page = await context.newPage();

  console.log("Navigating to http://localhost:3000...");
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.error('PAGE ERROR:', err.message));
  page.on('dialog', async dialog => {
    console.log(`PAGE DIALOG: [${dialog.type()}] "${dialog.message()}"`);
    await dialog.dismiss();
  });
  page.on('request', req => {
    if (req.url().includes('/api/')) {
      console.log(`REQ >> ${req.method()} ${req.url()}`);
    }
  });
  page.on('response', res => {
    if (res.url().includes('/api/')) {
      console.log(`RES << ${res.status()} ${res.url()}`);
    }
  });

  await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });

  // 1. Create group via custom modal
  console.log("Clicking 'Tạo nhóm mới' to open custom modal...");
  await page.click('button[title="Tạo nhóm mới"]');

  // Verify custom modal is visible
  await page.waitForSelector('h3:has-text("Tạo nhóm mới")');
  console.log("Custom Create Group Modal is visible.");

  // Type new group name and click submit button inside the modal
  const randomSuffix = Math.floor(Math.random() * 10000);
  const groupName = `Playwright Group ${randomSuffix}`;
  console.log(`Typing group name '${groupName}'...`);
  const inputSelector = 'input[placeholder="Nhập tên nhóm..."]';
  await page.click(inputSelector);
  await page.locator(inputSelector).pressSequentially(groupName, { delay: 50 });
  await page.waitForTimeout(1000);
  
  // Take debug screenshot of Create Group modal
  await page.screenshot({ path: '/Users/si/.gemini/antigravity-ide/brain/970e5e50-6513-44c5-b0fc-98a600e717c6/debug_create_group_modal.png' });
  console.log("Saved debug create group modal screenshot.");

  const inputValue = await page.$eval(inputSelector, el => el.value);
  console.log(`Value inside input field: "${inputValue}"`);

  await page.click('.fixed.inset-0 button:has-text("Tạo")');

  // Wait for dropdown to update and check active group
  await page.waitForTimeout(2000);
  const selectedGroupText = await page.$eval('select', el => {
    const val = el.value;
    const opt = Array.from(el.options).find(o => o.value === val);
    return opt ? opt.text : null;
  });
  console.log(`Active selected group in UI: "${selectedGroupText}"`);

  // 2. Upload verification
  console.log("Uploading test file verify_groups.cjs...");
  const fileInput = await page.$('#rag-file-upload');
  await fileInput.setInputFiles('/Users/si/Documents/Study/bieudientrithuc/graph_rag/FE/verify_groups.cjs');
  await page.waitForTimeout(4000); // Wait for file to list

  // 3. Delete document via custom modal
  console.log("Clicking delete document button...");
  await page.click('button[title="Xoá tài liệu"]');

  // Verify custom delete document modal is visible
  await page.waitForSelector('h3:has-text("Xóa tài liệu")');
  console.log("Custom Delete Document Modal is visible.");
  await page.click('div:has(h3:has-text("Xóa tài liệu")) button:has-text("Đồng ý xóa")');
  await page.waitForTimeout(2000);

  // 4. Delete group via custom modal
  console.log("Clicking delete group button...");
  await page.click('button[title="Xóa nhóm này"]');

  // Verify custom delete group modal is visible
  await page.waitForSelector('h3:has-text("Xóa nhóm tài liệu")');
  console.log("Custom Delete Group Modal is visible.");
  await page.click('div:has(h3:has-text("Xóa nhóm tài liệu")) button:has-text("Đồng ý xóa")');
  await page.waitForTimeout(2000);

  // Take screenshot of clean final state
  const screenshotPath = '/Users/si/.gemini/antigravity-ide/brain/970e5e50-6513-44c5-b0fc-98a600e717c6/custom_modals_verified.png';
  await page.screenshot({ path: screenshotPath });
  console.log(`Saved screenshot to: ${screenshotPath}`);

  await browser.close();
  console.log("Browser test completed successfully!");
}

run().catch(err => {
  console.error("Verification failed:", err);
  process.exit(1);
});
