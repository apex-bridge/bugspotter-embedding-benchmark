/**
 * Collect real SDK captures from the BugSpotter demo app.
 *
 * For each of the 25 bugs:
 *   1. Navigate to the buggy app in a real browser
 *   2. Trigger the bug (which causes a real error captured by the SDK)
 *   3. Intercept the POST /api/bugs request to capture the full report
 *   4. Save the raw capture (console logs, network requests, metadata)
 *
 * Usage:
 *   npm install playwright
 *   node data/collect_sdk_captures.js [--url https://demo.kz.bugspotter.io]
 *
 * Output:
 *   data/sdk-captures/raw/bug_01_checkout_crash.json
 *   data/sdk-captures/raw/bug_02_api_500.json
 *   ... (25 files)
 */

const { chromium } = require("playwright");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const http = require("http");

const DEFAULT_URL = "http://localhost:3000";
const BACKEND_MOCK_DIR = path.resolve(
  __dirname,
  "../../bugspotter-public/packages/backend-mock"
);
const DEMO_DIR = path.resolve(__dirname, "../../bugspotter-public/apps/demo");

// All 25 bugs: id, slug, trigger expression, component, error_type, wait time
const BUGS = [
  {
    id: 1,
    slug: "checkout_crash",
    trigger: "triggerCheckoutCrash()",
    component: "checkout",
    error_type: "js_error",
    wait: 2000,
  },
  {
    id: 2,
    slug: "api_500",
    trigger: "triggerApi500()",
    component: "product",
    error_type: "network_error",
    wait: 3000,
  },
  {
    id: 3,
    slug: "slow_page",
    trigger: "triggerSlowPage()",
    component: "orders",
    error_type: "performance",
    wait: 10000,
  },
  {
    id: 4,
    slug: "form_crash",
    trigger: "triggerFormCrash()",
    component: "checkout",
    error_type: "js_error",
    wait: 2000,
  },
  {
    id: 5,
    slug: "broken_image",
    trigger: "triggerBrokenImage()",
    component: "product",
    error_type: "network_error",
    wait: 3000,
  },
  {
    id: 6,
    slug: "rage_clicks",
    trigger: null, // special handling — click #rage-target 7 times
    component: "orders",
    error_type: "ui_interaction",
    wait: 3000,
  },
  {
    id: 7,
    slug: "json_parse_crash",
    trigger: "triggerJsonParseCrash()",
    component: "api",
    error_type: "js_error",
    wait: 2000,
  },
  {
    id: 8,
    slug: "stack_overflow",
    trigger: "triggerStackOverflow()",
    component: "feed",
    error_type: "js_error",
    wait: 2000,
  },
  {
    id: 9,
    slug: "null_dom",
    trigger: "triggerNullDom()",
    component: "profile",
    error_type: "js_error",
    wait: 2000,
  },
  {
    id: 10,
    slug: "array_bounds",
    trigger: "triggerArrayBounds()",
    component: "list",
    error_type: "js_error",
    wait: 2000,
  },
  {
    id: 11,
    slug: "invalid_date",
    trigger: "triggerInvalidDate()",
    component: "datepicker",
    error_type: "js_error",
    wait: 2000,
  },
  {
    id: 12,
    slug: "cors_error",
    trigger: "triggerCorsError()",
    component: "api",
    error_type: "network_error",
    wait: 8000,
  },
  {
    id: 13,
    slug: "network_timeout",
    trigger: "triggerNetworkTimeout()",
    component: "api",
    error_type: "network_error",
    wait: 8000,
  },
  {
    id: 14,
    slug: "unauthorized_401",
    trigger: "trigger401()",
    component: "auth",
    error_type: "network_error",
    wait: 3000,
  },
  {
    id: 15,
    slug: "rate_limit_429",
    trigger: "trigger429()",
    component: "api",
    error_type: "network_error",
    wait: 10000,
  },
  {
    id: 16,
    slug: "websocket_fail",
    trigger: "triggerWebSocketFail()",
    component: "websocket",
    error_type: "network_error",
    wait: 6000,
  },
  {
    id: 17,
    slug: "long_task",
    trigger: "triggerLongTask()",
    component: "feed",
    error_type: "performance",
    wait: 5000,
  },
  {
    id: 18,
    slug: "dom_explosion",
    trigger: "triggerDomExplosion()",
    component: "gallery",
    error_type: "performance",
    wait: 3000,
  },
  {
    id: 19,
    slug: "heavy_canvas",
    trigger: "triggerHeavyCanvas()",
    component: "upload",
    error_type: "performance",
    wait: 3000,
  },
  {
    id: 20,
    slug: "storage_quota",
    trigger: "triggerStorageQuota()",
    component: "cache",
    error_type: "js_error",
    wait: 2000,
  },
  {
    id: 21,
    slug: "stale_closure",
    trigger: "triggerStaleClosure()",
    component: "forms",
    error_type: "state_management",
    wait: 2000,
  },
  {
    id: 22,
    slug: "zindex_conflict",
    trigger: "triggerZIndex()",
    component: "table",
    error_type: "css_ui",
    wait: 2000,
  },
  {
    id: 23,
    slug: "layout_shift",
    trigger: "triggerLayoutShift()",
    component: "modal",
    error_type: "css_ui",
    wait: 10000,
  },
  {
    id: 24,
    slug: "unhandled_promise",
    trigger: "triggerUnhandledPromise()",
    component: "profile",
    error_type: "js_error",
    wait: 3000,
  },
  {
    id: 25,
    slug: "listener_leak",
    trigger: "triggerListenerLeak()",
    component: "list",
    error_type: "performance",
    wait: 2000,
  },
];

async function collectBug(page, bug, baseUrl, outputDir) {
  const label = `Bug ${String(bug.id).padStart(2, "0")}: ${bug.slug}`;
  console.log(`\n>>> ${label}`);

  // Navigate fresh for each bug (clean console/network state)
  await page.goto(`${baseUrl}/buggy-app.html`, { waitUntil: "networkidle" });

  // Wait for SDK to initialize
  await page.waitForFunction(() => typeof BugSpotter !== "undefined", {
    timeout: 10000,
  });
  await page.waitForTimeout(1000);

  // Intercept the SDK submit request
  let capturedReport = null;
  await page.route("**/api/v1/reports", async (route) => {
    const request = route.request();
    if (request.method() === "POST") {
      try {
        const body = JSON.parse(request.postData());
        capturedReport = body;
        console.log(`  Captured: "${body.title}"`);
      } catch (e) {
        console.log(`  Failed to parse request body: ${e.message}`);
      }
    }
    // Let the request continue (or fulfill with a mock response)
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify({
        success: true,
        data: { id: "bench-" + bug.id },
        timestamp: new Date().toISOString(),
      }),
    });
  });

  // Trigger the bug
  if (bug.trigger) {
    console.log(`  Triggering: ${bug.trigger}`);
    await page.evaluate(bug.trigger);
  } else if (bug.slug === "rage_clicks") {
    // Special handling: click the rage target 7 times rapidly
    console.log(`  Triggering: 7 rapid clicks on #rage-target`);
    const target = page.locator("#rage-target");
    for (let i = 0; i < 7; i++) {
      await target.click({ delay: 50 });
    }
  }

  // Wait for the SDK to capture and submit
  await page.waitForTimeout(bug.wait);

  // Remove the route interceptor
  await page.unroute("**/api/v1/reports");

  if (!capturedReport) {
    console.log(`  WARNING: No report captured for ${bug.slug}`);
    return false;
  }

  // Save raw capture with metadata
  const output = {
    bug_id: bug.id,
    bug_slug: bug.slug,
    component: bug.component,
    error_type: bug.error_type,
    original_title: capturedReport.title,
    original_description: capturedReport.description,
    report: capturedReport.report,
    captured_at: new Date().toISOString(),
  };

  const filename = `bug_${String(bug.id).padStart(2, "0")}_${bug.slug}.json`;
  const filepath = path.join(outputDir, filename);
  fs.writeFileSync(filepath, JSON.stringify(output, null, 2));
  console.log(`  Saved: ${filepath}`);

  // Log capture stats
  const r = capturedReport.report || {};
  const consoleCount = (r.console || []).length;
  const networkCount = (r.network || []).length;
  const hasScreenshot = !!(r.screenshot && r.screenshot !== "SCREENSHOT_FAILED");
  const replayEvents = (r.replay || []).length;
  console.log(
    `  Stats: ${consoleCount} console, ${networkCount} network, screenshot=${hasScreenshot}, ${replayEvents} replay events`
  );

  return true;
}

function waitForServer(url, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const check = () => {
      http
        .get(url, (res) => {
          if (res.statusCode < 500) resolve();
          else setTimeout(check, 500);
        })
        .on("error", () => {
          if (Date.now() - start > timeoutMs)
            reject(new Error(`Server at ${url} not ready after ${timeoutMs}ms`));
          else setTimeout(check, 500);
        });
    };
    check();
  });
}

function startServer(command, args, cwd, label) {
  console.log(`Starting ${label}...`);
  const proc = spawn(command, args, {
    cwd,
    stdio: "pipe",
    shell: true,
  });
  proc.stdout.on("data", (d) => {
    const line = d.toString().trim();
    if (line) console.log(`  [${label}] ${line}`);
  });
  proc.stderr.on("data", (d) => {
    const line = d.toString().trim();
    if (line && !line.includes("ExperimentalWarning"))
      console.log(`  [${label}] ${line}`);
  });
  return proc;
}

async function main() {
  const args = process.argv.slice(2);
  const urlIdx = args.indexOf("--url");
  const useExternal = urlIdx >= 0;
  const baseUrl = useExternal ? args[urlIdx + 1] : DEFAULT_URL;

  const outputDir = path.join(__dirname, "sdk-captures", "raw");
  fs.mkdirSync(outputDir, { recursive: true });

  console.log("=".repeat(60));
  console.log("BugSpotter SDK Capture Collection");
  console.log(`URL: ${baseUrl}`);
  console.log(`Output: ${outputDir}`);
  console.log(`Bugs: ${BUGS.length}`);
  console.log("=".repeat(60));

  // Start local servers if not using external URL
  let backendProc = null;
  let staticProc = null;

  if (!useExternal) {
    backendProc = startServer("node", ["server.js"], BACKEND_MOCK_DIR, "backend");
    await waitForServer("http://localhost:4000/health");
    console.log("Backend ready on :4000");

    staticProc = startServer(
      "npx",
      ["http-server", "-p", "3000", "-c-1", "--silent"],
      path.resolve(DEMO_DIR, "../.."),
      "static"
    );
    await waitForServer("http://localhost:3000/apps/demo/buggy-app.html");
    console.log("Static server ready on :3000");
  }

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    userAgent:
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
  });
  const page = await context.newPage();

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      console.log(`  [console.${msg.type()}] ${msg.text().substring(0, 120)}`);
    }
  });

  // For local mode, the buggy app is under /apps/demo/
  const appUrl = useExternal ? baseUrl : `${baseUrl}/apps/demo`;

  let succeeded = 0;
  let failed = 0;

  for (const bug of BUGS) {
    try {
      const ok = await collectBug(page, bug, appUrl, outputDir);
      if (ok) succeeded++;
      else failed++;
    } catch (error) {
      console.log(`  ERROR: ${error.message}`);
      failed++;
    }
  }

  await browser.close();

  // Stop local servers
  if (backendProc) backendProc.kill();
  if (staticProc) staticProc.kill();

  console.log("\n" + "=".repeat(60));
  console.log(`Done: ${succeeded} captured, ${failed} failed`);
  console.log(`Files: ${outputDir}`);
  console.log("=".repeat(60));
}

main().catch(console.error);
