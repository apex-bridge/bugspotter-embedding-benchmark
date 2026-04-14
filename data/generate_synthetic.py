"""
Generate synthetic bug reports and paraphrases for the embedding benchmark.

Source 2: 200 synthetic reports (20 archetypes x 10 variations each)
Source 3: 100 BugSpotter-native captures from a test SPA

Paraphrases are AGGRESSIVE — different structure, vocabulary, detail level —
so embedding models must understand semantics, not just word overlap.
"""

import json
import os
import random
import copy
import argparse
import re

# ---------------------------------------------------------------------------
# 20 Bug Archetypes — each defines a base bug report in BugSpotter format
# ---------------------------------------------------------------------------

ARCHETYPES = [
    {
        "archetype_id": "checkout_coupon",
        "title": "Checkout button unresponsive after coupon applied",
        "description": "After applying coupon code SAVE20, the Place Order button stops responding to clicks. No console errors. Works fine without coupon.",
        "console_logs": ["[warn] React: Cannot update during an existing state transition"],
        "stack_trace": None,
        "url": "/checkout",
        "browser": "Chrome 124",
        "error_type": "ui_interaction",
        "component": "checkout",
    },
    {
        "archetype_id": "login_csrf",
        "title": "Login fails with 403 Forbidden after session timeout",
        "description": "User gets a 403 error when submitting the login form after being idle for 30+ minutes. Refreshing the page fixes it. CSRF token seems to expire.",
        "console_logs": ["POST /api/auth/login 403 (Forbidden)", "Error: Request failed with status code 403"],
        "network_logs": [{"method": "POST", "url": "/api/auth/login", "status": 403, "duration": 45}],
        "stack_trace": None,
        "url": "/login",
        "browser": "Firefox 125",
        "error_type": "network_error",
        "component": "auth",
    },
    {
        "archetype_id": "infinite_scroll_memory",
        "title": "Memory leak on infinite scroll — page becomes unresponsive after scrolling",
        "description": "Feed page becomes extremely slow after scrolling through ~200 items. Chrome DevTools shows memory usage climbing from 80MB to 1.2GB. DOM nodes not being cleaned up.",
        "console_logs": ["[performance] Long task detected: 1250ms", "[warn] Possible memory leak: detached DOM nodes > 500"],
        "stack_trace": None,
        "url": "/feed",
        "browser": "Chrome 124",
        "error_type": "performance",
        "component": "feed",
    },
    {
        "archetype_id": "dark_mode_flash",
        "title": "White flash when navigating between pages in dark mode",
        "description": "In dark mode, there is a brief white flash during page transitions. The body background momentarily resets to white before the dark theme is applied.",
        "console_logs": [],
        "stack_trace": None,
        "url": "/settings",
        "browser": "Safari 17",
        "error_type": "css_ui",
        "component": "theme",
    },
    {
        "archetype_id": "file_upload_crash",
        "title": "App crashes when uploading files larger than 10MB",
        "description": "Dragging a 15MB PDF into the upload zone causes the entire page to freeze and eventually crash. Console shows an out-of-memory error in the FileReader callback.",
        "console_logs": ["Uncaught RangeError: Invalid array buffer length", "at FileReader.onload (upload.js:47:12)"],
        "stack_trace": "RangeError: Invalid array buffer length\n    at FileReader.onload (upload.js:47:12)\n    at processChunk (upload.js:32:5)",
        "url": "/documents/upload",
        "browser": "Chrome 124",
        "error_type": "js_error",
        "component": "upload",
    },
    {
        "archetype_id": "search_empty_state",
        "title": "Search shows 'No results' for a split second before actual results appear",
        "description": "When typing in the search bar, the 'No results found' message briefly flashes before results are rendered. This happens because the UI resets on each keystroke before the debounced search completes.",
        "console_logs": ["[warn] State update batching issue in SearchResults component"],
        "stack_trace": None,
        "url": "/search",
        "browser": "Chrome 124",
        "error_type": "state_management",
        "component": "search",
    },
    {
        "archetype_id": "modal_scroll_lock",
        "title": "Background page scrolls when modal is open on iOS",
        "description": "On iOS Safari, opening any modal dialog still allows the background page to scroll. The body scroll lock is not working on mobile WebKit.",
        "console_logs": [],
        "stack_trace": None,
        "url": "/dashboard",
        "browser": "Safari iOS 17",
        "error_type": "css_ui",
        "component": "modal",
    },
    {
        "archetype_id": "date_picker_timezone",
        "title": "Date picker shows wrong date for users in UTC+ timezones",
        "description": "Users in Asia/Tokyo timezone see the previous day selected in the date picker. The component uses new Date() without accounting for timezone offset.",
        "console_logs": ["[debug] DatePicker: selected=2024-03-14T15:00:00.000Z, display=March 14, expected=March 15"],
        "stack_trace": None,
        "url": "/events/create",
        "browser": "Chrome 124",
        "error_type": "js_error",
        "component": "datepicker",
    },
    {
        "archetype_id": "websocket_reconnect",
        "title": "Real-time notifications stop working after network reconnect",
        "description": "After losing WiFi and reconnecting, the WebSocket connection is not re-established. Users stop receiving real-time updates until they manually refresh the page.",
        "console_logs": ["WebSocket connection to 'wss://api.example.com/ws' failed", "Error: WebSocket is already in CLOSING or CLOSED state"],
        "network_logs": [{"method": "GET", "url": "wss://api.example.com/ws", "status": 502, "duration": 5000}],
        "stack_trace": None,
        "url": "/notifications",
        "browser": "Chrome 124",
        "error_type": "network_error",
        "component": "websocket",
    },
    {
        "archetype_id": "form_validation_race",
        "title": "Form submits with invalid data when clicking submit rapidly",
        "description": "Double-clicking the submit button on the registration form sends two requests. The second request goes through before client-side validation completes, resulting in a 422 error from the API.",
        "console_logs": ["POST /api/users 422 (Unprocessable Entity)", "[error] Validation failed: email already exists"],
        "network_logs": [{"method": "POST", "url": "/api/users", "status": 422, "duration": 120}],
        "stack_trace": None,
        "url": "/register",
        "browser": "Firefox 125",
        "error_type": "state_management",
        "component": "forms",
    },
    {
        "archetype_id": "image_lazy_load",
        "title": "Images fail to load when scrolling quickly in product gallery",
        "description": "In the product gallery, rapidly scrolling causes some images to remain as grey placeholders and never load. The IntersectionObserver seems to miss entries during fast scrolling.",
        "console_logs": ["[warn] IntersectionObserver: callback took 450ms", "GET /images/product-42.webp net::ERR_INSUFFICIENT_RESOURCES"],
        "stack_trace": None,
        "url": "/products",
        "browser": "Chrome 124",
        "error_type": "performance",
        "component": "gallery",
    },
    {
        "archetype_id": "ssr_hydration_mismatch",
        "title": "Hydration mismatch error on product detail page",
        "description": "Next.js throws a hydration mismatch error on the product detail page. Server renders the price as '$10.00' but client shows '$10' because of different number formatting between server and client.",
        "console_logs": ["Warning: Text content did not match. Server: \"$10.00\" Client: \"$10\"", "Error: Hydration failed because the initial UI does not match what was rendered on the server."],
        "stack_trace": "Error: Hydration failed\n    at throwOnHydrationMismatch (react-dom.js:12453)\n    at updateHostComponent (react-dom.js:14231)",
        "url": "/products/42",
        "browser": "Chrome 124",
        "error_type": "react_specific",
        "component": "product",
    },
    {
        "archetype_id": "cors_preflight",
        "title": "API calls fail with CORS error after deploying to new subdomain",
        "description": "All API requests from app.example.com to api.example.com fail with CORS error. The preflight OPTIONS request returns 405 Method Not Allowed. Backend CORS config only allows the old domain.",
        "console_logs": ["Access to XMLHttpRequest at 'https://api.example.com/data' from origin 'https://app.example.com' has been blocked by CORS policy",
                         "Response to preflight request doesn't pass access control check"],
        "network_logs": [{"method": "OPTIONS", "url": "https://api.example.com/data", "status": 405, "duration": 30},
                         {"method": "GET", "url": "https://api.example.com/data", "status": 0, "duration": 0}],
        "stack_trace": None,
        "url": "/dashboard",
        "browser": "Chrome 124",
        "error_type": "network_error",
        "component": "api",
    },
    {
        "archetype_id": "dropdown_z_index",
        "title": "Dropdown menu appears behind the table header on scroll",
        "description": "The action dropdown in each table row renders behind the sticky table header when the user scrolls down. z-index stacking context issue.",
        "console_logs": [],
        "stack_trace": None,
        "url": "/admin/users",
        "browser": "Chrome 124",
        "error_type": "css_ui",
        "component": "table",
    },
    {
        "archetype_id": "react_key_warning",
        "title": "List re-renders completely when a single item is added",
        "description": "Adding a new item to the todo list causes all items to re-render and lose their animation state. React DevTools shows missing key prop — using array index as key.",
        "console_logs": ["Warning: Each child in a list should have a unique \"key\" prop.",
                         "[perf] TodoList: 45 unnecessary re-renders detected"],
        "stack_trace": None,
        "url": "/todos",
        "browser": "Chrome 124",
        "error_type": "react_specific",
        "component": "list",
    },
    {
        "archetype_id": "oauth_redirect_loop",
        "title": "Infinite redirect loop after OAuth login with Google",
        "description": "After authenticating with Google, the user gets stuck in a redirect loop between /auth/callback and /login. The session cookie is not being set because SameSite=Strict blocks it on the redirect.",
        "console_logs": ["[warn] Cookie 'session' has been rejected because it has the 'SameSite=Strict' attribute but came from a cross-site response",
                         "Navigated to /login (redirected from /auth/callback)"],
        "network_logs": [{"method": "GET", "url": "/auth/callback", "status": 302, "duration": 15},
                         {"method": "GET", "url": "/login", "status": 302, "duration": 10}],
        "stack_trace": None,
        "url": "/auth/callback",
        "browser": "Chrome 124",
        "error_type": "network_error",
        "component": "auth",
    },
    {
        "archetype_id": "animation_jank",
        "title": "Accordion animation stutters on low-end Android devices",
        "description": "The expand/collapse animation for the FAQ accordion is extremely janky on budget Android phones. Profiling shows layout thrashing — height is being animated instead of transform.",
        "console_logs": ["[perf] Forced reflow detected during animation (15ms)", "[warn] Layout shift score: 0.32 (poor)"],
        "stack_trace": None,
        "url": "/faq",
        "browser": "Chrome Android 124",
        "error_type": "performance",
        "component": "accordion",
    },
    {
        "archetype_id": "stale_cache",
        "title": "Users see outdated dashboard data after deployment",
        "description": "After deploying a new version, some users see stale data on the dashboard. The service worker is caching API responses and not invalidating on version change.",
        "console_logs": ["[sw] Serving cached response for /api/dashboard/stats (age: 3600s)", "[warn] Service worker: skipWaiting() not called on activate"],
        "stack_trace": None,
        "url": "/dashboard",
        "browser": "Chrome 124",
        "error_type": "state_management",
        "component": "cache",
    },
    {
        "archetype_id": "a11y_focus_trap",
        "title": "Keyboard focus escapes dialog and moves to background elements",
        "description": "When tabbing through the settings dialog, focus moves to elements behind the modal overlay. Screen reader users can interact with background content while the modal is open.",
        "console_logs": ["[a11y] Focus trap violation: focus moved to element outside dialog boundary"],
        "stack_trace": None,
        "url": "/settings",
        "browser": "Firefox 125",
        "error_type": "css_ui",
        "component": "modal",
    },
    {
        "archetype_id": "useeffect_loop",
        "title": "useEffect infinite loop causes page to freeze on profile edit",
        "description": "Opening the profile edit form causes the page to freeze. React DevTools shows the component re-rendering thousands of times. A useEffect dependency array includes an object that gets recreated on each render.",
        "console_logs": ["Warning: Maximum update depth exceeded. This can happen when a component calls setState inside useEffect.",
                         "[error] Too many re-renders. React limits the number of renders to prevent an infinite loop."],
        "stack_trace": "Error: Too many re-renders\n    at renderWithHooks (react-dom.js:15486)\n    at mountIndeterminateComponent (react-dom.js:20103)\n    at beginWork (react-dom.js:21626)",
        "url": "/profile/edit",
        "browser": "Chrome 124",
        "error_type": "react_specific",
        "component": "profile",
    },
]

# ---------------------------------------------------------------------------
# AGGRESSIVE paraphrases — each is a completely different way to describe
# the same bug. Different words, different structure, different detail level.
# The archetype_id maps to a list of 9 paraphrase dicts.
# ---------------------------------------------------------------------------

DEEP_PARAPHRASES = {
    "checkout_coupon": [
        {"title": "Place Order btn dead after discount code", "description": "Applied SAVE20 coupon, now the order button is completely dead. No errors in console. Remove coupon => works again."},
        {"title": "Cannot complete purchase with promo code active", "description": "Whenever a promotional code is entered at checkout, the submit button becomes non-functional. No click handler fires."},
        {"title": "Order submission broken by coupon feature", "description": "The checkout flow breaks when users apply any discount code. Button appears clickable but nothing happens on click. Suspected state issue."},
        {"title": "Coupon breaks checkout CTA", "description": "Our checkout call-to-action stops working the moment a coupon is applied. Works in incognito without coupon. React state transition warning in console."},
        {"title": "klick on 'place order' does nothing with coupon", "description": "after entering cupon code the place order buton is not clickable anymore. tryed different browsers same result."},
        {"title": "Checkout regression: discount codes disable submit", "description": "Since last deploy, applying any discount code prevents form submission. The Place Order button receives clicks but the onClick handler never executes."},
        {"title": "Bug: impossible to order when coupon is used", "description": "Customers report they can't finish orders when using promo codes. I tested with SAVE20 and confirmed — the button just doesn't respond."},
        {"title": "Payment button frozen after promo applied", "description": "The payment submission button becomes unresponsive once a promotional discount is activated in the cart. Console shows React warning about state transitions."},
        {"title": "Оформление заказа не работает с купоном", "description": "After applying coupon SAVE20, Place Order button stops working. No console errors visible. Works without coupon applied."},
    ],
    "login_csrf": [
        {"title": "403 on login after being idle", "description": "If you leave the login page open for 30 min and then try to submit, you get a 403. Refreshing fixes it. Classic CSRF token expiry."},
        {"title": "Session timeout causes login form to break", "description": "The login form returns forbidden error when submitted after extended idle period. CSRF token in the form has expired server-side."},
        {"title": "Can't log in after leaving tab open", "description": "Left the browser open overnight, came back, typed credentials, hit submit — 403. Refresh and try again works fine."},
        {"title": "CSRF validation fails on stale login page", "description": "Login POST rejected with 403 when the page has been open longer than the CSRF token lifetime. Token rotation not implemented on the form."},
        {"title": "login broken if page left open too long", "description": "getting 403 forbiden when trying to login after leaving the page sitting there. have to reload first then it works."},
        {"title": "Authentication form rejects valid credentials periodically", "description": "Users intermittently report 403 errors on the login form. Root cause: CSRF tokens expire after 30 minutes but the form doesn't refresh them."},
        {"title": "Stale form token blocks authentication", "description": "The anti-CSRF mechanism is rejecting login attempts when the token has gone stale. Users who idle on the page hit this consistently."},
        {"title": "403 Forbidden: login form needs page reload after timeout", "description": "When the session expires while on /login, the embedded CSRF token becomes invalid. Next submit gets 403. UX fix: auto-refresh token."},
        {"title": "Login page returns 403 after long inactivity", "description": "The login endpoint rejects POST requests with 403 status after ~30min of inactivity because the CSRF token is no longer valid."},
    ],
    "infinite_scroll_memory": [
        {"title": "Page gets slower and slower when scrolling the feed", "description": "Scrolling through hundreds of feed items makes the page nearly unusable. Memory grows to 1+ GB. Old items are not being removed from DOM."},
        {"title": "Memory consumption grows unbounded in feed view", "description": "The feed page never releases DOM nodes from scrolled-past items. After 200+ items, Chrome reports over 1GB RSS and the tab becomes unresponsive."},
        {"title": "Browser tab crashes after extensive scrolling", "description": "If you keep scrolling the feed for a few minutes, Chrome eventually shows the 'Aw Snap' crash page. Memory profiler shows detached nodes piling up."},
        {"title": "Infinite scroll causes OOM on feed page", "description": "Virtualization is not working on the feed. Every loaded item stays in the DOM forever, causing out-of-memory after ~200 items."},
        {"title": "feed page very laggy after scrolling alot", "description": "the feed gets super slow after i scroll for a while. my computer fan goes crazy. i think its a memory problem because chrome uses 1gb+"},
        {"title": "Detached DOM nodes accumulate in infinite scroll list", "description": "Performance audit shows 500+ detached DOM nodes after scrolling the feed. No cleanup/virtualization is happening. Long tasks detected (1250ms+)."},
        {"title": "Feed performance degrades linearly with scroll depth", "description": "Each new batch of items is appended but nothing is removed. Page becomes unresponsive around item #200. Classic missing windowing implementation."},
        {"title": "Memory leak: feed items never garbage collected", "description": "DOM nodes from the infinite scroll feed are never cleaned up. Memory climbs from 80MB to 1.2GB steadily. Performance long task warnings in console."},
        {"title": "Бесконечная прокрутка вызывает утечку памяти", "description": "Feed page memory leak. Scrolling through ~200 items causes Chrome memory to jump from 80MB to 1.2GB. DOM nodes not cleaned up."},
    ],
    "dark_mode_flash": [
        {"title": "FOUC in dark theme during navigation", "description": "Brief flash of white background when moving between routes with dark mode enabled. The theme class is applied too late in the render cycle."},
        {"title": "Screen flickers white between pages (dark mode)", "description": "Users on dark mode see an annoying white flicker on every page change. Body bg resets momentarily before CSS applies the dark background."},
        {"title": "Dark mode: visible white gap during route change", "description": "There's a fraction-of-second white flash on SPA route transitions when dark theme is active. SSR is not preserving the theme state."},
        {"title": "Theme persistence broken on page navigation", "description": "The dark mode setting doesn't persist fast enough during navigation, causing a flash of the default white theme on each transition."},
        {"title": "annoying flashing when switching pages in dark mode", "description": "every time i click a link the screen goes white for a split second before dark mode kicks in. really bad for my eyes at night"},
        {"title": "Flash of unstyled content with dark color scheme", "description": "Route transitions cause the document to briefly render with the default light theme before the dark mode stylesheet takes effect."},
        {"title": "White background blink on dark mode transitions", "description": "During client-side navigation, the body element loses its dark theme background for approximately 50-100ms, creating a visible flash."},
        {"title": "Visual regression: dark theme flickers on navigate", "description": "Since the SPA migration, dark mode users see a white flash between pages. The theme context isn't available during the initial render of each route."},
        {"title": "Light theme bleeds through during page changes", "description": "When dark mode is on, there's a white flash during page transitions. Body background momentarily shows default before dark theme applies."},
    ],
    "file_upload_crash": [
        {"title": "Large file upload kills the browser tab", "description": "Uploading any file over 10MB freezes the page completely. The FileReader tries to load the entire file into memory at once."},
        {"title": "Page freezes on drag-and-drop of big PDFs", "description": "Dragging a 15MB file into the upload area causes Chrome to become unresponsive. RangeError in the console about invalid buffer length."},
        {"title": "FileReader OOM on large attachments", "description": "The file upload component crashes when processing files >10MB because FileReader.readAsArrayBuffer is called on the full file without chunking."},
        {"title": "Upload zone crashes browser with large documents", "description": "15MB+ files cause an out-of-memory crash in the upload handler. Stack trace points to FileReader.onload at upload.js:47."},
        {"title": "cant upload big files, page freezez", "description": "tried uploading a 15mb pdf and the whole page froze. had to close the tab. smaller files work ok tho"},
        {"title": "Out of memory error in file upload callback", "description": "RangeError: Invalid array buffer length thrown at FileReader.onload when uploading files exceeding ~10MB. No chunked upload implementation."},
        {"title": "Document upload crashes on files exceeding 10MB", "description": "The upload feature doesn't handle large files properly. No streaming or chunking — tries to buffer the entire file, causing memory overflow."},
        {"title": "Browser crash: RangeError in upload.js during big file upload", "description": "Uploading a large file (tested with 15MB PDF) triggers RangeError: Invalid array buffer length at upload.js:47. Page becomes unresponsive."},
        {"title": "Upload component breaks with large files", "description": "Dragging a 15MB PDF into the upload zone causes page to freeze and crash. FileReader callback at upload.js:47 throws RangeError."},
    ],
    "search_empty_state": [
        {"title": "Empty state flickers during search input", "description": "The 'no results' placeholder briefly appears between every keystroke while searching. Debounce and state management issue."},
        {"title": "Search results flash empty state on each keystroke", "description": "Typing in search causes the empty state component to render for ~100ms before real results appear. State resets on each input event."},
        {"title": "Visual glitch: search results disappear and reappear", "description": "While typing a query, results vanish momentarily showing 'Nothing found' before the actual matches render. Very distracting UX."},
        {"title": "Search UX broken — flashing empty state", "description": "The SearchResults component resets to empty on every keystroke, then re-renders with results after the debounced API call returns."},
        {"title": "search bar shows no results message while typing", "description": "when i type in search it keeps flashing 'no results found' between each letter. the results show up but the flickering is annoying"},
        {"title": "Race condition in search results rendering", "description": "State batching issue causes the search UI to briefly show the empty state between debounced queries. Each keystroke clears results prematurely."},
        {"title": "Debounced search shows intermediate empty state", "description": "The search component clears its result state on input change, then waits for the debounced query. This creates a visible flicker of the empty state."},
        {"title": "SearchResults component: batching issue causes flicker", "description": "React state update batching problem in SearchResults — the component shows 'No results' for a frame before displaying actual matches on each keystroke."},
        {"title": "Flicker in search — empty state shows between results", "description": "Brief flash of 'No results found' message between keystrokes in search. State resets before debounced search returns. Batching issue."},
    ],
    "modal_scroll_lock": [
        {"title": "iOS: page scrolls behind open modal", "description": "Modal overlay doesn't prevent background scrolling on iOS Safari. The body scroll lock CSS technique doesn't work on mobile WebKit."},
        {"title": "Touch scrolling passes through modal on iPhone", "description": "On iPhones, you can scroll the page content behind the modal by swiping. Standard overflow:hidden on body doesn't work in mobile Safari."},
        {"title": "Body scroll not locked when dialog is open (mobile Safari)", "description": "The modal component fails to lock body scroll on iOS. Users can scroll the background content while the dialog is displayed."},
        {"title": "Scroll leak through modal overlay on mobile", "description": "Modal dialogs don't properly block background scrolling on iOS Safari. overflow:hidden on body is ignored by mobile WebKit."},
        {"title": "can still scroll the page behind popups on iphone", "description": "when i open any popup/modal on my iphone i can still scroll the background page. it works fine on desktop"},
        {"title": "Modal scroll containment fails on iOS WebKit", "description": "Body scroll lock implementation (overflow:hidden) is ineffective on iOS Safari. Background content remains scrollable when modal is active."},
        {"title": "Background scrollable during modal on iOS devices", "description": "The body scroll prevention technique doesn't work in mobile Safari. Modal overlays allow background content scrolling on all iOS devices."},
        {"title": "iOS Safari ignores overflow:hidden when modal is shown", "description": "Standard body scroll lock via CSS overflow:hidden doesn't work on iOS Safari. Need touch-action:none or position:fixed workaround for modals."},
        {"title": "Scroll lock broken on mobile Safari", "description": "On iOS Safari, opening any modal dialog still allows background page scrolling. Body scroll lock not working on mobile WebKit."},
    ],
    "date_picker_timezone": [
        {"title": "Wrong date displayed in datepicker for non-UTC users", "description": "Users east of UTC see yesterday's date pre-selected. The date component doesn't handle timezone offsets correctly."},
        {"title": "Date off by one day in Asian timezones", "description": "Tokyo/Seoul users report the date picker shows the previous day. The ISO date string conversion doesn't account for local timezone."},
        {"title": "Timezone bug in date selection component", "description": "new Date() without timezone awareness causes the datepicker to show wrong dates for UTC+ users. Off by one day for Asia/Tokyo."},
        {"title": "Calendar shows yesterday for users in UTC+9", "description": "The event creation form pre-selects the wrong date for users in Japan/Korea. Date math doesn't account for timezone offset."},
        {"title": "date picker off by one day for some users", "description": "some of our users in japan are seeing the wrong date in the date picker. it shows yesterday instead of today"},
        {"title": "Date component: UTC conversion bug for positive offset timezones", "description": "The DatePicker uses new Date() which returns UTC. For UTC+ timezones, the displayed date is one day behind the user's local date."},
        {"title": "Off-by-one date for timezones ahead of UTC", "description": "Date picker displays March 14 for a user in Asia/Tokyo when it should show March 15. Raw JS Date objects used without timezone conversion."},
        {"title": "Datepicker timezone handling: UTC+ users see previous day", "description": "new Date() returns UTC time. For users in positive UTC offsets, the date picker renders the wrong calendar day (one behind)."},
        {"title": "Wrong day selected in date picker for Japan timezone", "description": "Users in UTC+9 timezone see previous day in date picker. Component uses new Date() without timezone offset."},
    ],
    "websocket_reconnect": [
        {"title": "WebSocket doesn't reconnect after WiFi drops", "description": "Losing network and reconnecting leaves the WebSocket dead. No auto-reconnect logic. Users must refresh to get real-time updates."},
        {"title": "Live updates break after network interruption", "description": "When internet connection is lost and restored, the WebSocket stays disconnected. The notification feed stops updating in real-time."},
        {"title": "No auto-reconnect for realtime notification socket", "description": "The WebSocket connection handler doesn't implement reconnection. After any network disruption, real-time features stop until page reload."},
        {"title": "Notifications stop after losing internet briefly", "description": "A brief WiFi dropout permanently kills the WebSocket connection. No reconnection strategy implemented. Real-time notifications go silent."},
        {"title": "live notifications stop working when wifi reconnects", "description": "if my wifi drops for a second the live updates stop completely. i have to refresh the page to get them back"},
        {"title": "WebSocket CLOSED state not handled on network recovery", "description": "After network reconnection, the WebSocket remains in CLOSED state. Error: 'WebSocket is already in CLOSING or CLOSED state'. No retry logic."},
        {"title": "Real-time feed dies after connectivity blip", "description": "Any network interruption permanently breaks the WebSocket. The client doesn't attempt reconnection. Users lose all real-time functionality."},
        {"title": "Missing WebSocket reconnection handler", "description": "The WebSocket client has no logic to detect disconnection and reconnect. After WiFi drops, the connection stays dead until manual page refresh."},
        {"title": "WebSocket connection not re-established after network loss", "description": "After losing WiFi and reconnecting, WebSocket stays disconnected. Users stop receiving real-time notifications until page refresh."},
    ],
    "form_validation_race": [
        {"title": "Double submit sends two API requests", "description": "Clicking the register button twice quickly fires two POST requests. Server processes both, second one fails with duplicate email error."},
        {"title": "Race condition on form submit allows invalid data", "description": "Rapid clicking of submit bypasses client-side validation. The second request reaches the server before the first validation completes."},
        {"title": "Registration form can be submitted multiple times", "description": "No submit debouncing on the signup form. Fast double-click sends duplicate requests, causing 422 errors for duplicate entries."},
        {"title": "Double-click on register creates duplicate API calls", "description": "The registration form doesn't disable the submit button after click. Two rapid clicks = two POST /api/users requests = 422 on the second."},
        {"title": "clicking submit fast sends the form twice", "description": "if you double click the register button it sends two requests and the second one fails with email already exists error"},
        {"title": "Submit button lacks debounce — allows duplicate requests", "description": "The registration form's submit handler doesn't prevent multiple rapid submissions. This causes race conditions and 422 validation errors."},
        {"title": "No double-submit protection on signup form", "description": "Users can trigger multiple form submissions by clicking quickly. The button is never disabled during the async POST request, allowing duplicates."},
        {"title": "Client validation race: form sent before checks complete", "description": "Rapid submit clicks bypass the async email validation. The second request fires before the first validation response arrives, causing 422."},
        {"title": "Form double-submit causes 422 error on registration", "description": "Double-clicking submit on registration form sends two requests. Second request fails with 422 because email already exists from first."},
    ],
    "image_lazy_load": [
        {"title": "Product images stuck as grey placeholders on fast scroll", "description": "Quickly scrolling the product gallery leaves many images permanently unloaded. The lazy loading observer misses them entirely."},
        {"title": "Lazy-loaded images never appear after rapid scrolling", "description": "Fast scrolling through the gallery causes IntersectionObserver to miss elements. Images remain as grey boxes even after scrolling stops."},
        {"title": "Gallery images don't load on quick scroll-through", "description": "If you scroll fast past product images, some never load — they stay as grey placeholders forever. Observer callback too slow."},
        {"title": "IntersectionObserver misses images during fast scroll", "description": "The product gallery's lazy loading breaks during rapid scrolling. Observer callbacks take 450ms+, missing fast-scrolled-past elements."},
        {"title": "images dont load when scrolling fast in product page", "description": "when i scroll fast thru the products page some images just stay grey and never load. slow scrolling works fine"},
        {"title": "Lazy loading race: images stuck in placeholder state", "description": "Product gallery images fail to load during rapid scrolling. IntersectionObserver callback takes too long and entries are missed."},
        {"title": "Product gallery: some images permanently stuck loading", "description": "After fast scrolling, certain product images remain as placeholders. The IntersectionObserver loses track of elements that pass through quickly."},
        {"title": "ERR_INSUFFICIENT_RESOURCES on gallery fast scroll", "description": "Rapid scrolling triggers too many concurrent image requests. Some fail with net::ERR_INSUFFICIENT_RESOURCES and never retry."},
        {"title": "Grey placeholder images persist after fast scrolling", "description": "Images fail to load when scrolling quickly in product gallery. IntersectionObserver misses entries during fast scrolling."},
    ],
    "ssr_hydration_mismatch": [
        {"title": "Next.js hydration error: server/client price mismatch", "description": "Product page shows hydration warning. Server formats price as $10.00 but client renders $10. Number formatting differs between environments."},
        {"title": "SSR hydration fails on product prices", "description": "Server-side rendered price doesn't match client-side format. React throws 'Text content did not match' error. Intl.NumberFormat inconsistency."},
        {"title": "Price rendering differs between server and client", "description": "Hydration mismatch on product detail page. The server renders '$10.00' while the client hydrates with '$10'. Different toLocaleString behavior."},
        {"title": "React hydration warning on product detail page", "description": "Next.js reports text content mismatch for price display. Server: '$10.00', Client: '$10'. Currency formatter behaves differently in SSR vs browser."},
        {"title": "product page throws hydration error in console", "description": "getting 'text content did not match' errors on the product page. the price shows differently from what the server sent"},
        {"title": "Server/client content divergence in price component", "description": "The price formatter produces different output on server vs client, triggering React hydration failure. Server: $10.00, Browser: $10."},
        {"title": "Hydration failed: number format inconsistency in SSR", "description": "Product detail page hydration error. The Intl.NumberFormat produces '$10.00' on the server but '$10' on the client due to locale differences."},
        {"title": "Next.js: throwOnHydrationMismatch for price display", "description": "react-dom.js:12453 throws hydration mismatch. Price component renders differently between SSR and CSR. Currency formatting is not isomorphic."},
        {"title": "Price format mismatch between server and client rendering", "description": "Hydration mismatch error on product page. Server renders '$10.00', client shows '$10'. Different number formatting between environments."},
    ],
    "cors_preflight": [
        {"title": "CORS blocks all API requests from new subdomain", "description": "Migrated frontend to app.example.com, now every API call fails with CORS. The server only allows the old origin in Access-Control-Allow-Origin."},
        {"title": "Preflight OPTIONS returns 405 after domain change", "description": "API requests fail because the CORS preflight check gets a 405 response. Backend hasn't been updated to allow the new frontend domain."},
        {"title": "Cross-origin requests blocked after subdomain migration", "description": "All fetch calls from the new app subdomain are blocked by CORS. The preflight request fails because the server rejects OPTIONS with 405."},
        {"title": "CORS policy error on every API call", "description": "Since moving to the new subdomain, Chrome blocks all API requests. The Access-Control-Allow-Origin header doesn't include app.example.com."},
        {"title": "api calls broken after moving to new domain, cors error", "description": "we moved the frontend to a new subdomain and now all api calls fail with cors error. the backend needs to be updated with the new origin"},
        {"title": "Access-Control-Allow-Origin not updated for new frontend URL", "description": "After deploying frontend to new subdomain, all API calls fail CORS preflight. Server still has old domain in allowed origins list."},
        {"title": "405 on preflight: CORS config stale after migration", "description": "The backend's CORS configuration still references the old frontend domain. OPTIONS requests return 405, blocking all cross-origin API calls."},
        {"title": "All XHR blocked: CORS policy doesn't allow new origin", "description": "Every API request from app.example.com gets blocked because api.example.com's CORS headers only whitelist the previous frontend URL."},
        {"title": "CORS error on all API requests after subdomain change", "description": "API calls from app.example.com to api.example.com blocked by CORS. Preflight OPTIONS returns 405. Backend config needs new domain."},
    ],
    "dropdown_z_index": [
        {"title": "Action menu hidden behind sticky header", "description": "The row action dropdown renders underneath the sticky table header when scrolled. z-index issue with stacking contexts."},
        {"title": "Dropdown clipped by sticky table header on scroll", "description": "Table action menus get hidden behind the sticky header when the page is scrolled. Need to adjust the z-index stacking."},
        {"title": "Context menu invisible under fixed header", "description": "Row-level dropdown menus in the admin table are rendered behind the sticky header element after scrolling. Stacking context problem."},
        {"title": "Table row actions obscured by sticky header", "description": "The action dropdown for each row appears behind the sticky table header once the user scrolls down. z-index conflict."},
        {"title": "dropdown menu goes behind the header when i scroll down", "description": "in the users admin page when you scroll down and click the action menu it shows behind the sticky header. cant see the options"},
        {"title": "Stacking context bug: popover behind sticky element", "description": "The table row action popover/dropdown has a lower stacking order than the sticky header. After scrolling, it renders behind the header."},
        {"title": "z-index conflict: action dropdown vs sticky <thead>", "description": "Admin table's per-row dropdown gets overlapped by the sticky table header. The dropdown's parent has a lower z-index stacking context."},
        {"title": "CSS: dropdown z-index lower than sticky header", "description": "Dropdown menus in table rows render behind the position:sticky table header on scroll. Need to create a new stacking context for the dropdown."},
        {"title": "Dropdown menu renders behind sticky table header", "description": "Action dropdown in table row appears behind sticky header when scrolled. z-index stacking context issue."},
    ],
    "react_key_warning": [
        {"title": "All todo items re-render when adding new item", "description": "The entire list re-renders and animations reset whenever a new item is added. Array index being used as key instead of stable ID."},
        {"title": "Missing key prop causes full list re-render", "description": "Adding a todo causes all existing items to lose animation state and re-render. React DevTools confirms index-based keys."},
        {"title": "Performance issue: list uses index as key", "description": "Each item addition triggers a complete re-render of all items. Using array index as React key means React can't tell items apart."},
        {"title": "React key warning + excessive re-renders in todo list", "description": "Console shows 'Each child should have unique key prop'. Profiler shows 45 unnecessary re-renders when one item is added."},
        {"title": "todo list rerenders everything when you add one item", "description": "adding a new todo makes all the existing ones flash/rerender. the animations all restart too. react is complaining about missing keys"},
        {"title": "Index-based keys cause animation state loss in list", "description": "Todo list uses array index as key prop. Adding items causes React to re-mount all subsequent items, losing their internal state and animations."},
        {"title": "Unnecessary re-renders: 45 components re-render for 1 addition", "description": "React DevTools shows massive unnecessary re-renders in TodoList. Root cause: missing stable key prop, using array index instead."},
        {"title": "List component performance: needs unique keys", "description": "Todo items lose animation state on every add/remove because the list uses array index as the key prop. React remounts all items."},
        {"title": "Complete list re-render on single item add in todo", "description": "Adding new todo item causes all items to re-render and lose animation state. Using array index as key. React key warning in console."},
    ],
    "oauth_redirect_loop": [
        {"title": "Google OAuth: infinite loop between callback and login", "description": "After Google authentication, user bounces between /auth/callback and /login endlessly. SameSite=Strict cookie isn't set on cross-site redirect."},
        {"title": "Endless redirect after Google sign-in", "description": "Completing Google OAuth results in a redirect loop. The session cookie can't be set due to SameSite restrictions on the OAuth callback."},
        {"title": "OAuth callback fails to set session — redirect loop", "description": "The OAuth redirect from Google doesn't set the session cookie because of SameSite=Strict policy. User ends up in login→callback→login loop."},
        {"title": "Google auth: SameSite cookie blocks session creation", "description": "OAuth flow breaks because SameSite=Strict prevents the session cookie from being set during the cross-origin redirect from Google."},
        {"title": "stuck in redirect loop after google login", "description": "after logging in with google the page keeps bouncing between login and callback pages. never actually logs in. works with email/password tho"},
        {"title": "Cross-site cookie rejection breaks OAuth flow", "description": "SameSite=Strict attribute on the session cookie causes it to be rejected during the Google OAuth callback redirect. Results in auth loop."},
        {"title": "Session not persisted after OAuth: SameSite policy issue", "description": "The authentication loop occurs because the session cookie with SameSite=Strict is silently dropped on the cross-site OAuth redirect."},
        {"title": "Cookie SameSite=Strict incompatible with OAuth redirect", "description": "Google OAuth redirect back to /auth/callback can't set the session cookie (SameSite=Strict blocks it). User enters redirect loop."},
        {"title": "Redirect loop after Google OAuth authentication", "description": "After Google auth, stuck in loop between /auth/callback and /login. Session cookie not set because SameSite=Strict blocks cross-site."},
    ],
    "animation_jank": [
        {"title": "FAQ accordion choppy on cheap phones", "description": "Expand/collapse animations are extremely stuttery on low-end Android. The component animates height instead of using GPU-accelerated transform."},
        {"title": "Janky animation: accordion uses height instead of transform", "description": "FAQ accordion causes layout thrashing on each frame because it animates the height property. Profiling shows forced reflow every 15ms."},
        {"title": "Layout thrashing in accordion expand animation", "description": "The FAQ section's accordion triggers forced reflows during animation because it changes element height. Very visible jank on budget devices."},
        {"title": "Accordion animation: poor CLS on mobile", "description": "The expand/collapse animation in the FAQ causes a cumulative layout shift of 0.32. It should use transform:scaleY instead of height transition."},
        {"title": "faq accordion is super laggy on my android phone", "description": "the faq expand/collapse animation is really choppy on my budget android. on desktop it looks smooth"},
        {"title": "Forced reflow during CSS height animation", "description": "FAQ accordion animates max-height which triggers layout recalculation on every frame. Shows 15ms forced reflows. Use transform instead."},
        {"title": "Poor performance: accordion animates layout property", "description": "The FAQ accordion transitions height (a layout property) instead of a composite-only property like transform. Causes jank on slow GPUs."},
        {"title": "CLS 0.32 from accordion: needs GPU-accelerated animation", "description": "Layout shift score of 0.32 during FAQ accordion animation. Height transitions cause layout thrashing. Should use transform or opacity."},
        {"title": "Accordion animation stutters on low-end devices", "description": "FAQ accordion is janky on budget Android. Profiling shows layout thrashing from animating height. CLS: 0.32."},
    ],
    "stale_cache": [
        {"title": "Dashboard shows old data after deploy", "description": "Some users see yesterday's dashboard stats after a new deployment. Service worker caches API responses and doesn't clear cache on update."},
        {"title": "Service worker serves stale API responses", "description": "The service worker's cache-first strategy is serving outdated dashboard data. Cache invalidation on new version deployment is missing."},
        {"title": "Outdated numbers on dashboard after new release", "description": "Post-deployment, users report seeing stale statistics. The service worker continues serving cached API responses from the previous version."},
        {"title": "SW cache not invalidated on deploy", "description": "The service worker doesn't call skipWaiting() on activate, so old cache is used. Dashboard shows stale data until user manually refreshes."},
        {"title": "dashboard data is wrong after update, shows old numbers", "description": "after the latest deploy some users see old data on the dashboard. clearing browser cache fixes it. service worker issue?"},
        {"title": "Stale service worker cache persists across deployments", "description": "Dashboard API responses cached by the SW aren't invalidated when a new version deploys. skipWaiting() not called on activate event."},
        {"title": "Cache-first SW strategy breaks on redeploy", "description": "Service worker continues serving cached /api/dashboard/stats (3600s old) after deployment because the activation handler doesn't invalidate the cache."},
        {"title": "Service worker: skipWaiting not called, cache persists", "description": "The SW install event caches API responses but the activate event doesn't call skipWaiting() or clear old caches. Users see stale data."},
        {"title": "Users see stale dashboard data after new deployment", "description": "After deploying new version, users see old dashboard stats. Service worker caches API responses and doesn't invalidate on version change."},
    ],
    "a11y_focus_trap": [
        {"title": "Tab key escapes modal dialog to background", "description": "Tabbing through the settings dialog sends focus to elements behind the modal. No focus trap implemented for screen reader accessibility."},
        {"title": "Focus not contained within modal overlay", "description": "Users navigating with keyboard can tab out of the dialog into background elements. Violates WCAG 2.4.3 focus order requirements."},
        {"title": "Screen reader can access content behind open modal", "description": "Assistive technology users can interact with background page while modal is open because focus isn't trapped inside the dialog."},
        {"title": "Missing focus trap in settings dialog", "description": "The settings modal doesn't implement a focus trap. Keyboard users can tab to background elements that should be inert when dialog is open."},
        {"title": "can tab to elements behind the popup with keyboard", "description": "when using tab key in the settings popup focus goes to buttons and links behind the overlay. accessibility problem"},
        {"title": "WCAG violation: dialog doesn't trap keyboard focus", "description": "The settings dialog lacks a focus trap implementation. Tab key cycles to elements outside the dialog boundary when modal is displayed."},
        {"title": "Accessibility: focus escapes dialog boundary", "description": "Focus management issue in modal component. Keyboard navigation (Tab) is not constrained to the dialog, allowing interaction with background."},
        {"title": "a11y focus trap missing on modal component", "description": "The modal overlay doesn't trap focus. Screen reader and keyboard users can access and interact with content behind the open dialog."},
        {"title": "Focus escapes dialog to background elements", "description": "When tabbing through settings dialog, focus moves to elements behind modal overlay. No focus trap for keyboard/screen reader users."},
    ],
    "useeffect_loop": [
        {"title": "Infinite re-render on profile edit form", "description": "The profile edit page freezes the browser. useEffect with object dependency triggers on every render because the object reference changes."},
        {"title": "useEffect dependency causes infinite render loop", "description": "Profile edit component re-renders infinitely because the useEffect dependency array contains an inline object that's recreated each render."},
        {"title": "Browser hangs on profile edit — React render loop", "description": "Opening profile edit page hangs the browser. React error: 'Maximum update depth exceeded'. useEffect has unstable dependency."},
        {"title": "setState in useEffect with object dep = infinite loop", "description": "The profile form's useEffect calls setState, which creates a new object, which triggers the effect again. Classic React infinite loop."},
        {"title": "profile edit page freezes browser completely", "description": "when i open the edit profile page my browser freezes. console says too many re-renders. something about useeffect"},
        {"title": "Maximum update depth exceeded in profile component", "description": "React throws 'Maximum update depth exceeded' on /profile/edit. The useEffect has an object in its dependency array that's recreated on each render cycle."},
        {"title": "React infinite loop: useEffect with non-memoized dep", "description": "Profile edit form creates an infinite render loop. The effect depends on an object literal that changes identity every render, re-triggering the effect."},
        {"title": "Profile page: useEffect → setState → re-render → loop", "description": "The profile edit component hits React's re-render limit. Root cause: useEffect dependency is an object literal (new reference each render)."},
        {"title": "useEffect infinite loop on profile edit — page freezes", "description": "Opening profile edit form freezes the page. Component re-renders thousands of times. useEffect dependency array has object recreated each render."},
    ],
}


# ---------------------------------------------------------------------------
# Noise injection — makes some reports more realistic
# ---------------------------------------------------------------------------

def add_typos(text, rate=0.03):
    """Randomly introduce typos at the given rate."""
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < rate and chars[i].isalpha():
            mutation = random.choice(["swap", "drop", "double"])
            if mutation == "swap" and i + 1 < len(chars):
                chars[i], chars[i+1] = chars[i+1], chars[i]
            elif mutation == "drop":
                chars[i] = ""
            elif mutation == "double":
                chars[i] = chars[i] * 2
    return "".join(chars)


def truncate_description(text, keep_ratio=0.5):
    """Simulate an incomplete bug report."""
    sentences = text.split(". ")
    keep = max(1, int(len(sentences) * keep_ratio))
    return ". ".join(sentences[:keep])


# ---------------------------------------------------------------------------
# BugSpotter-native captures — simulated SDK output from a test SPA
# These share components/errors with archetypes to create HARD negatives
# ---------------------------------------------------------------------------

BUGSPOTTER_CAPTURES = [
    # These bugs are in the SAME components as archetypes but are DIFFERENT bugs
    # This creates natural hard negatives (D3)
    {"id": "bs_000", "title": "Checkout total shows NaN after removing last item",
     "description": "Remove the only item from cart while on checkout page. The total displays NaN. The price calculation doesn't handle empty cart edge case.",
     "console_logs": ["TypeError: Cannot read properties of undefined (reading 'price')"],
     "stack_trace": "TypeError: Cannot read properties of undefined (reading 'price')\n    at calculateTotal (checkout.js:89:22)",
     "url": "/checkout", "browser": "Chrome 124", "error_type": "js_error", "component": "checkout",
     "group": "bs_group_checkout_nan"},
    {"id": "bs_001", "title": "Checkout NaN bug when cart becomes empty",
     "description": "If you remove all items from the cart on the checkout page, the order total shows NaN instead of $0.00. Price calculation crashes on empty array.",
     "console_logs": ["TypeError: Cannot read properties of undefined (reading 'price')"],
     "stack_trace": None, "url": "/checkout", "browser": "Firefox 125", "error_type": "js_error", "component": "checkout",
     "group": "bs_group_checkout_nan"},

    {"id": "bs_002", "title": "Login form accepts empty password field",
     "description": "The login form can be submitted with an empty password. Client-side validation doesn't check for empty strings, only for null/undefined.",
     "console_logs": ["POST /api/auth/login 422 (Unprocessable Entity)"],
     "stack_trace": None, "url": "/login", "browser": "Chrome 124", "error_type": "state_management", "component": "auth",
     "group": "bs_group_login_validation"},
    {"id": "bs_003", "title": "Empty password passes client validation on login",
     "description": "Can submit the login form without entering a password. The required field check only validates against null, not empty string.",
     "console_logs": ["POST /api/auth/login 422 (Unprocessable Entity)"],
     "stack_trace": None, "url": "/login", "browser": "Safari 17", "error_type": "state_management", "component": "auth",
     "group": "bs_group_login_validation"},

    {"id": "bs_004", "title": "Feed crashes when post contains invalid UTF-8",
     "description": "A post with emoji sequences crashes the feed rendering. The text truncation function splits a multi-byte character, creating invalid UTF-8.",
     "console_logs": ["Uncaught DOMException: Failed to execute 'createTextNode': invalid character"],
     "stack_trace": "DOMException: Failed to execute 'createTextNode'\n    at FeedItem.render (feed.js:156:12)",
     "url": "/feed", "browser": "Chrome 124", "error_type": "js_error", "component": "feed",
     "group": "bs_group_feed_utf8"},
    {"id": "bs_005", "title": "Feed rendering breaks on posts with certain emojis",
     "description": "Some posts with complex emoji cause the feed to crash. The string slice function cuts in the middle of a surrogate pair.",
     "console_logs": ["Uncaught DOMException: Failed to execute 'createTextNode': invalid character"],
     "stack_trace": None, "url": "/feed", "browser": "Firefox 125", "error_type": "js_error", "component": "feed",
     "group": "bs_group_feed_utf8"},

    {"id": "bs_006", "title": "Dark mode toggle doesn't persist after page refresh",
     "description": "Toggling dark mode works visually but the preference is lost on refresh. The setting is stored in component state but not persisted to localStorage.",
     "console_logs": [],
     "stack_trace": None, "url": "/settings", "browser": "Chrome 124", "error_type": "state_management", "component": "theme",
     "group": "bs_group_theme_persist"},
    {"id": "bs_007", "title": "Theme preference resets on page reload",
     "description": "Dark mode selection doesn't survive a page refresh. The theme toggle updates React state but never writes to localStorage or cookie.",
     "console_logs": [],
     "stack_trace": None, "url": "/settings", "browser": "Edge 124", "error_type": "state_management", "component": "theme",
     "group": "bs_group_theme_persist"},

    {"id": "bs_008", "title": "Upload progress bar stuck at 0% for small files",
     "description": "Files under 1MB show 0% progress until completion, then jump to 100%. The progress event doesn't fire frequently enough for small uploads.",
     "console_logs": ["[warn] Upload progress event count: 1 (expected >= 5)"],
     "stack_trace": None, "url": "/documents/upload", "browser": "Chrome 124", "error_type": "ui_interaction", "component": "upload",
     "group": "bs_group_upload_progress"},
    {"id": "bs_009", "title": "Small file upload: progress goes from 0% straight to 100%",
     "description": "The upload progress indicator doesn't update for files under 1MB. It stays at 0% then jumps to 100%. Missing intermediate progress events.",
     "console_logs": ["[warn] Upload progress event count: 1 (expected >= 5)"],
     "stack_trace": None, "url": "/documents/upload", "browser": "Safari 17", "error_type": "ui_interaction", "component": "upload",
     "group": "bs_group_upload_progress"},

    {"id": "bs_010", "title": "Search suggestions don't close when clicking outside",
     "description": "The search autocomplete dropdown stays open when clicking anywhere outside it. Missing click-outside handler on the suggestions popover.",
     "console_logs": [],
     "stack_trace": None, "url": "/search", "browser": "Chrome 124", "error_type": "ui_interaction", "component": "search",
     "group": "bs_group_search_dropdown"},
    {"id": "bs_011", "title": "Autocomplete popup doesn't dismiss on outside click",
     "description": "Clicking outside the search suggestions dropdown doesn't close it. The component has no useClickOutside hook or onBlur handler.",
     "console_logs": [],
     "stack_trace": None, "url": "/search", "browser": "Firefox 125", "error_type": "ui_interaction", "component": "search",
     "group": "bs_group_search_dropdown"},

    {"id": "bs_012", "title": "Modal close button is invisible on light backgrounds",
     "description": "The X button in modals is white-on-white when the modal content has a light background. The button color doesn't adapt to content theme.",
     "console_logs": [],
     "stack_trace": None, "url": "/dashboard", "browser": "Chrome 124", "error_type": "css_ui", "component": "modal",
     "group": "bs_group_modal_close"},
    {"id": "bs_013", "title": "Can't see modal dismiss button — contrast issue",
     "description": "The modal's close icon has insufficient contrast against light modal backgrounds. WCAG AA contrast ratio not met.",
     "console_logs": ["[a11y] Color contrast ratio 1.2:1 below minimum 4.5:1"],
     "stack_trace": None, "url": "/dashboard", "browser": "Chrome 124", "error_type": "css_ui", "component": "modal",
     "group": "bs_group_modal_close"},

    {"id": "bs_014", "title": "Date picker doesn't close when selecting a date",
     "description": "After clicking a date in the picker, the dropdown stays open. User has to click outside to dismiss it. Expected: auto-close on selection.",
     "console_logs": [],
     "stack_trace": None, "url": "/events/create", "browser": "Chrome 124", "error_type": "ui_interaction", "component": "datepicker",
     "group": "bs_group_datepicker_close"},
    {"id": "bs_015", "title": "Calendar dropdown remains open after date selection",
     "description": "Selecting a date in the calendar picker doesn't auto-close the dropdown. Missing onSelect callback to toggle the popover state.",
     "console_logs": [],
     "stack_trace": None, "url": "/events/create", "browser": "Safari 17", "error_type": "ui_interaction", "component": "datepicker",
     "group": "bs_group_datepicker_close"},

    {"id": "bs_016", "title": "WebSocket floods server with reconnect attempts",
     "description": "When the WS server is down, the client retries connection every 100ms without backoff. This creates thousands of failed connections per minute.",
     "console_logs": ["WebSocket connection to 'wss://api.example.com/ws' failed", "[error] Reconnect attempt #4521"],
     "network_logs": [{"method": "GET", "url": "wss://api.example.com/ws", "status": 502, "duration": 100}],
     "stack_trace": None, "url": "/notifications", "browser": "Chrome 124", "error_type": "network_error", "component": "websocket",
     "group": "bs_group_ws_flood"},
    {"id": "bs_017", "title": "WS reconnect loop: no exponential backoff",
     "description": "The WebSocket client's reconnect logic has no backoff strategy. Retries every 100ms, DDoS-ing our own server when it's temporarily down.",
     "console_logs": ["WebSocket connection to 'wss://api.example.com/ws' failed"],
     "network_logs": [{"method": "GET", "url": "wss://api.example.com/ws", "status": 502, "duration": 100}],
     "stack_trace": None, "url": "/notifications", "browser": "Firefox 125", "error_type": "network_error", "component": "websocket",
     "group": "bs_group_ws_flood"},

    {"id": "bs_018", "title": "Register form doesn't show password strength indicator",
     "description": "The registration form accepts any password without showing strength feedback. Users can set '123' as password with no warning.",
     "console_logs": [],
     "stack_trace": None, "url": "/register", "browser": "Chrome 124", "error_type": "ui_interaction", "component": "forms",
     "group": "bs_group_register_strength"},
    {"id": "bs_019", "title": "No password strength validation on signup",
     "description": "Password field on registration accepts anything including single characters. No strength meter or minimum requirements enforced on client side.",
     "console_logs": [],
     "stack_trace": None, "url": "/register", "browser": "Edge 124", "error_type": "ui_interaction", "component": "forms",
     "group": "bs_group_register_strength"},

    # Remaining captures to reach ~100, with duplicates pairs
    {"id": "bs_020", "title": "Product gallery arrows don't work on touch devices",
     "description": "The prev/next arrows in the product image carousel don't respond to taps on mobile. Click events not handled for touch.",
     "console_logs": [], "stack_trace": None, "url": "/products", "browser": "Safari iOS 17", "error_type": "ui_interaction", "component": "gallery",
     "group": "bs_group_gallery_touch"},
    {"id": "bs_021", "title": "Carousel navigation broken on mobile — arrows unresponsive",
     "description": "Product image gallery prev/next buttons don't work with touch input. Desktop click works fine. Missing touch event listeners.",
     "console_logs": [], "stack_trace": None, "url": "/products", "browser": "Chrome Android 124", "error_type": "ui_interaction", "component": "gallery",
     "group": "bs_group_gallery_touch"},

    {"id": "bs_022", "title": "Product filter count shows wrong number",
     "description": "The 'showing X of Y products' counter doesn't update when filters are applied. Always shows the total count regardless of active filters.",
     "console_logs": [], "stack_trace": None, "url": "/products", "browser": "Chrome 124", "error_type": "state_management", "component": "product",
     "group": "bs_group_product_count"},
    {"id": "bs_023", "title": "Filter badge count incorrect after applying product filters",
     "description": "Active filter count badge still shows total products instead of filtered count. The count state isn't updated when filter changes.",
     "console_logs": [], "stack_trace": None, "url": "/products", "browser": "Firefox 125", "error_type": "state_management", "component": "product",
     "group": "bs_group_product_count"},

    {"id": "bs_024", "title": "CORS error when loading user avatar from CDN",
     "description": "User profile images fail to load from CDN with CORS error. The CDN doesn't have Access-Control-Allow-Origin set for our domain.",
     "console_logs": ["Access to image at 'https://cdn.example.com/avatars/user.jpg' from origin 'https://app.example.com' has been blocked by CORS policy"],
     "network_logs": [{"method": "GET", "url": "https://cdn.example.com/avatars/user.jpg", "status": 0, "duration": 0}],
     "stack_trace": None, "url": "/profile", "browser": "Chrome 124", "error_type": "network_error", "component": "api",
     "group": "bs_group_cdn_cors"},
    {"id": "bs_025", "title": "Profile avatars blocked by CDN CORS policy",
     "description": "Avatar images don't load because the CDN hasn't been configured with proper CORS headers for our app domain. Shows CORS blocked error.",
     "console_logs": ["Access to image blocked by CORS policy: No 'Access-Control-Allow-Origin' header"],
     "network_logs": [{"method": "GET", "url": "https://cdn.example.com/avatars/user.jpg", "status": 0, "duration": 0}],
     "stack_trace": None, "url": "/profile", "browser": "Safari 17", "error_type": "network_error", "component": "api",
     "group": "bs_group_cdn_cors"},

    {"id": "bs_026", "title": "Sticky table header misaligned after column resize",
     "description": "Resizing table columns causes the sticky header cells to become misaligned with the body cells. The widths go out of sync.",
     "console_logs": [], "stack_trace": None, "url": "/admin/users", "browser": "Chrome 124", "error_type": "css_ui", "component": "table",
     "group": "bs_group_table_align"},
    {"id": "bs_027", "title": "Column widths don't match between header and body in table",
     "description": "After manually resizing columns, the header row and body rows have different column widths. Sticky positioning breaks the width sync.",
     "console_logs": [], "stack_trace": None, "url": "/admin/users", "browser": "Firefox 125", "error_type": "css_ui", "component": "table",
     "group": "bs_group_table_align"},

    {"id": "bs_028", "title": "React StrictMode double-render causes duplicate API calls",
     "description": "In development, every component mounts twice due to StrictMode. This causes duplicate API calls on every page, doubling server load during dev.",
     "console_logs": ["[warn] Duplicate API call detected: GET /api/users (2 calls in 50ms)"],
     "stack_trace": None, "url": "/dashboard", "browser": "Chrome 124", "error_type": "react_specific", "component": "api",
     "group": "bs_group_strictmode"},
    {"id": "bs_029", "title": "Double API requests in dev mode — StrictMode side effect",
     "description": "React StrictMode mounts components twice, triggering useEffect API calls twice. Server sees duplicate requests for every data fetch.",
     "console_logs": ["[warn] Duplicate API call detected: GET /api/users"],
     "stack_trace": None, "url": "/dashboard", "browser": "Chrome 124", "error_type": "react_specific", "component": "api",
     "group": "bs_group_strictmode"},

    {"id": "bs_030", "title": "useEffect cleanup not called on profile navigation",
     "description": "Navigating away from profile edit doesn't trigger useEffect cleanup. The interval timer keeps running, updating state on an unmounted component.",
     "console_logs": ["Warning: Can't perform a React state update on an unmounted component"],
     "stack_trace": None, "url": "/profile/edit", "browser": "Chrome 124", "error_type": "react_specific", "component": "profile",
     "group": "bs_group_profile_cleanup"},
    {"id": "bs_031", "title": "Memory leak: timer continues after leaving profile page",
     "description": "The profile page starts a setInterval that's never cleared. After navigating away, React warns about state updates on unmounted component.",
     "console_logs": ["Warning: Can't perform a React state update on an unmounted component"],
     "stack_trace": None, "url": "/profile/edit", "browser": "Firefox 125", "error_type": "react_specific", "component": "profile",
     "group": "bs_group_profile_cleanup"},

    # More captures for variety — single reports (no duplicate pair)
    {"id": "bs_032", "title": "Tooltip stays visible after element is scrolled out of view",
     "description": "Hovering over a table cell shows a tooltip, but scrolling the table while tooltip is shown leaves it floating in the wrong position.",
     "console_logs": [], "stack_trace": None, "url": "/admin/users", "browser": "Chrome 124", "error_type": "css_ui", "component": "table",
     "group": "bs_group_tooltip_scroll"},

    {"id": "bs_033", "title": "Notification bell count doesn't decrement on read",
     "description": "The unread notification badge count stays the same even after opening and reading notifications. The count is only refreshed on page load.",
     "console_logs": [], "stack_trace": None, "url": "/notifications", "browser": "Chrome 124", "error_type": "state_management", "component": "websocket",
     "group": "bs_group_notif_count"},

    {"id": "bs_034", "title": "Auth token not refreshed before expiry",
     "description": "The JWT token expires and API calls start failing with 401 before the refresh logic kicks in. There's a gap between token expiry and refresh.",
     "console_logs": ["GET /api/data 401 (Unauthorized)", "[error] Token expired at 14:00, refresh attempted at 14:01"],
     "network_logs": [{"method": "GET", "url": "/api/data", "status": 401, "duration": 25}],
     "stack_trace": None, "url": "/dashboard", "browser": "Chrome 124", "error_type": "network_error", "component": "auth",
     "group": "bs_group_token_refresh"},

    {"id": "bs_035", "title": "Form loses all input on browser back button",
     "description": "Pressing browser back from step 2 of the wizard clears all data entered in step 1. Form state isn't persisted in history.",
     "console_logs": [], "stack_trace": None, "url": "/onboarding", "browser": "Safari 17", "error_type": "state_management", "component": "forms",
     "group": "bs_group_form_history"},

    {"id": "bs_036", "title": "Loading spinner never stops on slow connections",
     "description": "On 3G connections, the dashboard loading spinner runs indefinitely. The data fetch has no timeout, so it waits forever for a response.",
     "console_logs": ["[warn] Fetch /api/dashboard/stats pending for 45000ms"],
     "network_logs": [{"method": "GET", "url": "/api/dashboard/stats", "status": 0, "duration": 45000}],
     "stack_trace": None, "url": "/dashboard", "browser": "Chrome Android 124", "error_type": "network_error", "component": "cache",
     "group": "bs_group_loading_timeout"},

    {"id": "bs_037", "title": "Accordion state not preserved on page re-render",
     "description": "Expanding an FAQ item then switching tabs and coming back collapses all items. Open/closed state is only in component state, not URL.",
     "console_logs": [], "stack_trace": None, "url": "/faq", "browser": "Chrome 124", "error_type": "state_management", "component": "accordion",
     "group": "bs_group_accordion_state"},

    {"id": "bs_038", "title": "List pagination breaks when filtering and paginating simultaneously",
     "description": "Applying a filter while on page 3 of the list shows empty results. The page number isn't reset to 1 when filters change.",
     "console_logs": [], "stack_trace": None, "url": "/admin/users", "browser": "Chrome 124", "error_type": "state_management", "component": "list",
     "group": "bs_group_list_pagination"},

    {"id": "bs_039", "title": "OAuth popup blocked by browser on mobile",
     "description": "The Google OAuth popup is blocked by mobile browsers' popup blocker. The popup is triggered asynchronously after an await, losing the user gesture context.",
     "console_logs": ["[warn] Popup blocked: not triggered by user gesture"],
     "stack_trace": None, "url": "/login", "browser": "Safari iOS 17", "error_type": "network_error", "component": "auth",
     "group": "bs_group_oauth_popup"},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_archetype_report(archetype: dict, variation_idx: int) -> dict:
    """Generate a single report from an archetype + variation index."""
    aid = archetype["archetype_id"]
    group = f"syn_group_{aid}"

    if variation_idx == 0:
        # Original
        report = copy.deepcopy(archetype)
        report["id"] = f"syn_{aid}_00"
        report["group"] = group
        report.pop("archetype_id", None)
        report.pop("component", None)
        return report

    # Paraphrase from DEEP_PARAPHRASES
    paraphrases = DEEP_PARAPHRASES.get(aid, [])
    if variation_idx - 1 < len(paraphrases):
        para = paraphrases[variation_idx - 1]
    else:
        # Fallback: use original with noise
        para = {"title": archetype["title"], "description": archetype["description"]}

    report = {
        "id": f"syn_{aid}_{variation_idx:02d}",
        "title": para["title"],
        "description": para["description"],
        "console_logs": archetype.get("console_logs", []),
        "network_logs": archetype.get("network_logs", []),
        "stack_trace": archetype.get("stack_trace"),
        "url": archetype["url"],
        "browser": archetype["browser"],
        "error_type": archetype["error_type"],
        "group": group,
    }

    # Add noise to some variations
    if variation_idx in (4, 5):  # The "typo" variations
        report["title"] = add_typos(report["title"], rate=0.04)
    if variation_idx == 8:  # Truncated
        report["description"] = truncate_description(report["description"])
    if variation_idx % 3 == 0:
        report["console_logs"] = []  # Some reporters don't include logs

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic bug reports")
    parser.add_argument("--output", default="data/bug_reports.json",
                        help="Output path for combined dataset")
    parser.add_argument("--github-input", default="data/github_issues.json",
                        help="Path to scraped GitHub issues")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_reports = []

    # Source 1: GitHub issues (already scraped)
    if os.path.exists(args.github_input):
        with open(args.github_input, "r", encoding="utf-8") as f:
            github_reports = json.load(f)
        print(f"Loaded {len(github_reports)} GitHub issues from {args.github_input}")
        all_reports.extend(github_reports[:300])
    else:
        print(f"WARNING: {args.github_input} not found. Run scrape_github.py first.")

    # Source 2: Synthetic bug reports (20 archetypes x 10 variations)
    synthetic_reports = []
    for archetype in ARCHETYPES:
        for v in range(10):  # 0 = original, 1-9 = paraphrases
            report = generate_archetype_report(archetype, v)
            synthetic_reports.append(report)

    print(f"Generated {len(synthetic_reports)} synthetic reports (20 archetypes x 10 variations)")
    all_reports.extend(synthetic_reports)

    # Source 3: BugSpotter-native captures
    print(f"Generated {len(BUGSPOTTER_CAPTURES)} BugSpotter-native captures")
    all_reports.extend(BUGSPOTTER_CAPTURES)

    # Save combined dataset
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(all_reports)} bug reports saved to {args.output}")


if __name__ == "__main__":
    main()
