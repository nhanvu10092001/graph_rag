const { chromium } = require('@playwright/test');
const path = require('path');

async function run() {
  console.log("Launching browser...");
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 }
  });
  const page = await context.newPage();

  // Register dialog handler for the prompt window
  page.on('dialog', async dialog => {
    console.log(`[Dialog] Type: ${dialog.type()}, Message: "${dialog.message()}"`);
    if (dialog.type() === 'prompt') {
      await dialog.accept('AI Group');
      console.log("[Dialog] Accepted prompt with value: 'AI Group'");
    } else {
      await dialog.accept();
    }
  });

  console.log("Navigating to http://localhost:3000...");
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
  console.log("Page loaded. Title:", await page.title());

  // Click create group button
  console.log("Clicking 'Tạo nhóm mới' button...");
  await page.click('button[title="Tạo nhóm mới"]');
  await page.waitForTimeout(3000);

  // Select the newly created group in the dropdown (it's the second option, i.e., value of the newly created group)
  await page.evaluate(() => {
    const select = document.querySelector('select');
    if (select && select.options.length > 1) {
      select.selectedIndex = 1;
      select.dispatchEvent(new Event('change', { bubbles: true }));
    }
  });
  await page.waitForTimeout(1000);

  // Check selected value of dropdown
  const selectedGroupText = await page.$eval('select', el => {
    const val = el.value;
    const opt = Array.from(el.options).find(o => o.value === val);
    return opt ? opt.text : null;
  });
  console.log(`Active selected group in UI: "${selectedGroupText}"`);

  // Upload file
  console.log("Uploading graph_group_fact.txt...");
  const fileInput = await page.$('#rag-file-upload');
  await fileInput.setInputFiles('/Users/si/Documents/Study/bieudientrithuc/graph_rag/FE/graph_group_fact.txt');

  console.log("Waiting for file to be indexed...");
  await page.waitForTimeout(15000);

  // Check status in UI
  const statusTexts = await page.evaluate(() => {
    const badges = Array.from(document.querySelectorAll('span'));
    return badges.map(b => b.innerText).filter(t => t === 'INDEXED' || t === 'PROCESSING' || t === 'FAILED');
  });
  console.log("Document status badges in UI:", statusTexts);

  // Click "+ Tạo hội thoại mới" to clean chat area before sending
  console.log("Clicking 'Tạo hội thoại mới'...");
  await page.click('button:has-text("Tạo hội thoại mới")');
  await page.waitForTimeout(2000);

  // Send message
  const promptText = "Who created the Antigravity assistant?";
  console.log(`Sending message: "${promptText}"`);
  await page.fill('#chat-input-textarea', promptText);
  await page.keyboard.press('Enter');
  
  console.log("Waiting for streaming response (20s)...");
  await page.waitForTimeout(20000);

  // Print all messages
  const messages = await page.evaluate(() => {
    const bubbles = Array.from(document.querySelectorAll('div.p-4.rounded-2xl'));
    return bubbles.map(b => b.innerText);
  });
  console.log("Messages in chat window:", messages);

  // Save screenshot
  const screenshotPath = '/Users/si/.gemini/antigravity-ide/brain/970e5e50-6513-44c5-b0fc-98a600e717c6/groups_e2e_verified.png';
  await page.screenshot({ path: screenshotPath });
  console.log(`Saved screenshot to: ${screenshotPath}`);

  await browser.close();
  console.log("Browser closed successfully.");
}

run().catch(err => {
  console.error("Verification failed:", err);
  process.exit(1);
});
