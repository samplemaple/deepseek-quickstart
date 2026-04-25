/* GEO内容提取脚本 */
(function(){
  var _o = window.__GEO_ORIGIN__ || '';
  if (!_o) { alert('\u914d\u7f6e\u9519\u8bef'); return; }

  if (location.hostname.indexOf('xiaohongshu') === -1 && location.hostname.indexOf('xhslink') === -1) {
    alert('\u8bf7\u5728\u5c0f\u7ea2\u4e66\u7b14\u8bb0\u9875\u9762\u4f7f\u7528');
    return;
  }

  var _t = '', _d = '', _imgs = [];

  // 提取标题
  var _te = document.querySelector('#detail-title') || document.querySelector('[class*="title"]');
  if (_te) _t = (_te.innerText || '').trim();
  if (!_t) { var _m = document.querySelector('meta[property="og:title"]'); if (_m) _t = (_m.content || '').trim(); }

  // 提取正文
  var _de = document.querySelector('#detail-desc') || document.querySelector('[class*="desc"]') || document.querySelector('[class*="content"]');
  if (_de) _d = (_de.innerText || '').trim();
  if (!_d) { var _m2 = document.querySelector('meta[property="og:description"]'); if (_m2) _d = (_m2.content || '').trim(); }

  // 提取话题标签
  var _tags = [];
  document.querySelectorAll('a[href*="/search_result/"]').forEach(function(a) {
    var txt = (a.innerText || '').trim();
    if (txt && txt.startsWith('#')) _tags.push(txt);
  });
  if (_tags.length > 0) _d = _d + '\n\n' + _tags.join(' ');

  // 提取图片
  document.querySelectorAll('img').forEach(function(img) {
    var src = img.src || img.getAttribute('data-src') || '';
    if (src && (src.indexOf('xhscdn') !== -1 || src.indexOf('sns-img') !== -1) && src.indexOf('avatar') === -1 && _imgs.indexOf(src) === -1) {
      if (img.naturalWidth > 100 || img.width > 100 || !img.naturalWidth) _imgs.push(src);
    }
  });
  if (_imgs.length === 0) { var _ogi = document.querySelector('meta[property="og:image"]'); if (_ogi && _ogi.content) _imgs.push(_ogi.content); }

  if (!_t && !_d) { alert('\u672a\u627e\u5230\u7b14\u8bb0\u5185\u5bb9'); return; }

  var _tip = document.createElement('div');
  _tip.style.cssText = 'position:fixed;top:20px;right:20px;z-index:99999;background:#ff4757;color:#fff;padding:12px 20px;border-radius:8px;font-size:14px;box-shadow:0 4px 12px rgba(0,0,0,0.3)';
  _tip.textContent = '\u2705 \u63d0\u53d6\u6210\u529f\uff0c\u6b63\u5728\u8df3\u8f6c...';
  document.body.appendChild(_tip);

  var _xhr = new XMLHttpRequest();
  _xhr.open('POST', _o + '/api/v1/client-extract', true);
  _xhr.setRequestHeader('Content-Type', 'application/json');
  _xhr.onload = function() {
    try {
      var _r = JSON.parse(_xhr.responseText);
      if (_r.success && _r.token) {
        var _url = _o + '/?token=' + _r.token;
        // 先尝试直接跳转（不会被拦截）
        _tip.innerHTML = '\u2705 \u63d0\u53d6\u6210\u529f\uff01<br><a href="' + _url + '" target="_blank" style="color:#fff;text-decoration:underline;font-size:16px">\u70b9\u51fb\u8fd9\u91cc\u8df3\u8f6c\u5230\u4e8c\u521b\u5de5\u5177 \u2192</a>';
        _tip.style.cssText = 'position:fixed;top:20px;right:20px;z-index:99999;background:#2ed573;color:#fff;padding:16px 24px;border-radius:8px;font-size:14px;box-shadow:0 4px 12px rgba(0,0,0,0.3);line-height:1.8';
        // 3秒后自动跳转
        setTimeout(function(){ location.href = _url; }, 2000);
      } else {
        alert('\u63d0\u4ea4\u5931\u8d25: ' + (_r.error || ''));
        _tip.remove();
      }
    } catch(e) { alert('\u5f02\u5e38: ' + e.message); _tip.remove(); }
  };
  _xhr.onerror = function() { alert('\u7f51\u7edc\u9519\u8bef'); _tip.remove(); };
  _xhr.send(JSON.stringify({ title: _t, text: _d, image_urls: _imgs, source_url: location.href }));
})();
